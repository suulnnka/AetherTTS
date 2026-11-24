#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a4) 1 分钟深度诊断与 ChunkAttn 提速校验脚本。

直接复用 RTX 4090 上已有的 `./aethertts_exp` 产物（E3 best.pt、CausalVocosLite、特征缓存），
无需重训即可一次性定位「听不清 / UTMOS 低 / 电音」的精确环节，并实测零 chunk 循环 ChunkAttn 的等价性与提速倍率：

  A. 全部模型训练历史 (history.json) 收敛趋势与尾段斜率
  B. Train vs Val 各分项 Loss（含归一化与原始 log-mel L1 及 dB 误差估算）
  C. Aligner 深度体检（attn_soft 锐度/有效宽度、dur_gt vs dur_pred 相关性、dur==1 塌缩率、逐音素对齐表 + 对齐热力图 PNG）
  D. 前端一致性审计（训练缓存 BZNSYP prosody_pinyin vs 推理 pypinyin 差异与 <unk> 率）
  E. 8 链路逐级隔离评测（含最关键的 teacher-forced dur_gt + GT/Pred F0 -> 预训练 Vocos；自动走 CPU int8 Whisper 出 CER/MER + UTMOS + DTW Raw Mel L1 + 导出试听 WAV）
  F. ChunkLookaheadBackbone 零循环并行 vs 逐块串行 数值/梯度等价校验 + GPU/CPU 提速实测

用法：
    python scripts/diagnose_phase1.py --root ./aethertts_exp
    python scripts/diagnose_phase1.py --root ./aethertts_exp --model E3_chunk_attn_C4_L8 --num-utts 5
    python scripts/diagnose_phase1.py --root ./aethertts_exp --whisper-model large-v3-turbo
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import sys
import tarfile
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1_SCRIPT = REPO_ROOT / "scripts" / "aethertts_phase1.py"


def load_phase1_defs(storage_root: Path):
    """从 scripts/aethertts_phase1.py 提取 Cell 3 (前端/Mel/F0) 与 Cell 5 (模型/对齐/声码器) 定义，不触发下载或训练。"""
    src = PHASE1_SCRIPT.read_text(encoding="utf-8")
    m_start = re.search(r"^# Cell 3:", src, flags=re.M)
    m_end = re.search(r"^# 6b\. 向量化热路径", src, flags=re.M)
    if not (m_start and m_end and m_end.start() > m_start.start()):
        raise RuntimeError(f"无法在 {PHASE1_SCRIPT} 中定位 Cell 3..Cell 5 代码段，请确认脚本已同步。")
    block = src[m_start.start() : m_end.start()]
    ns = {
        "__name__": "_aether_diag_defs",
        "__file__": str(PHASE1_SCRIPT),
        "os": os, "sys": sys, "json": json, "math": math, "time": time,
        "signal": signal, "shutil": shutil, "random": __import__("random"),
        "tarfile": tarfile, "warnings": warnings, "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor, "torch": torch,
        "ENV_MODE": "local",
        "STORAGE_ROOT": storage_root,
        "LOCAL_WORK": storage_root,
        "CFG": {
            "seed": 42, "sample_rate": 24000, "n_fft": 1024, "hop_length": 256,
            "win_length": 1024, "n_mels": 100, "profile": "standard_phase1",
            "dataset_source": "bznsyp",
        },
    }
    exec(compile(block, f"{PHASE1_SCRIPT.name}#cells_3_5", "exec"), ns)
    return ns


def find_checkpoint(root: Path, model_name: str, profile: str, dataset: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        assert p.exists(), f"指定 checkpoint 不存在: {p}"
        return p
    runs_dir = root / "runs"
    cands = sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_*")) if runs_dir.exists() else []
    if not cands and runs_dir.exists():
        cands = sorted(runs_dir.glob(f"{model_name}*"))
    for d in reversed(cands):
        for fn in ("best.pt", "latest.pt"):
            if (d / fn).exists():
                return d / fn
    raise FileNotFoundError(f"在 {runs_dir} 中未找到 {model_name} 的 best.pt / latest.pt，请通过 --ckpt 指定。")


def find_vocoder_ckpt(root: Path, profile: str, dataset: str, explicit: str = ""):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    voc_dir = root / "vocoder"
    if not voc_dir.exists():
        return None
    cands = sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_*")) or sorted(voc_dir.glob("causal_vocos_lite_*"))
    for d in reversed(cands):
        if (d / "final.pt").exists():
            return d / "final.pt"
    return None


def find_manifest(root: Path, dataset: str, max_hours: float = None, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p / "manifest.json" if p.is_dir() else p
    cache_dir = root / "cache"
    if max_hours is not None:
        exact = cache_dir / f"{dataset}_v2_max{max_hours}h" / "manifest.json"
        if exact.exists():
            return exact
    cands = sorted(cache_dir.glob(f"{dataset}_v2_max*h/manifest.json")) if cache_dir.exists() else []
    if not cands and cache_dir.exists():
        cands = sorted(cache_dir.glob("*/manifest.json"))
    if not cands:
        raise FileNotFoundError(f"在 {cache_dir} 中未找到 manifest.json，请通过 --cache-dir 指定。")
    return cands[-1]


def resolve_feat_path(entry: dict, manifest_dir: Path) -> Path:
    p = Path(entry["feat_path"])
    if p.exists():
        return p
    alt = manifest_dir / p.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"特征文件不存在: {p} / {alt}")


def dtw_mel_l1(pred_mel: np.ndarray, gt_mel: np.ndarray) -> float:
    """对不同长度的原始 log-mel (n_mels, F1) 与 (n_mels, F2) 做 DTW 对齐后求逐 bin L1，消除语速/时长偏移假象。"""
    F1, F2 = pred_mel.shape[-1], gt_mel.shape[-1]
    # 帧间平均 L1 距离矩阵: (F1, F2)
    cost = np.abs(pred_mel.T[:, None, :] - gt_mel.T[None, :, :]).mean(axis=-1)
    dp = np.full((F1, F2), np.inf, dtype=np.float64)
    dp[0, 0] = cost[0, 0]
    for i in range(1, F1):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, F2):
        dp[0, j] = dp[0, j - 1] + cost[0, j]
    for i in range(1, F1):
        for j in range(1, F2):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    # 回溯路径长度用于求均值
    i, j, steps = F1 - 1, F2 - 1, 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin([dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]]))
            if k == 0:
                i -= 1; j -= 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        steps += 1
    return float(dp[F1 - 1, F2 - 1] / max(steps, 1))


def edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def compute_cer(ref: str, hyp: str) -> float:
    def tok(s):
        s = unicodedata.normalize("NFKC", s).lower()
        return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", s)
    r, h = tok(ref), tok(hyp)
    return 0.0 if not r else float(edit_distance(r, h)) / len(r)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def section_a_history(root: Path):
    print("\n" + "=" * 78)
    print("[A] 全部模型训练历史收敛轨迹 (runs/*/history.json)")
    print("=" * 78)
    runs_dir = root / "runs"
    hfiles = sorted(runs_dir.glob("*/history.json")) if runs_dir.exists() else []
    if not hfiles:
        print("  (未找到 history.json)")
        return {}
    out = {}
    print(f"  {'Run 目录':46s} | {'步数范围':13s} | {'val_mel 轨迹 (首->中->末)':26s} | {'尾段Δ':7s} | {'val_dur':7s} | {'bos%':5s}")
    print("  " + "-" * 118)
    for hf in hfiles:
        try:
            h = json.loads(hf.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = h.get("steps", [])
        vm = h.get("val_mel", [])
        vd = h.get("val_dur", [])
        vb = h.get("val_bos_ratio", [])
        if not steps or not vm:
            continue
        mid = len(vm) // 2
        tail_delta = (vm[-1] - vm[max(0, len(vm) - 3)]) if len(vm) >= 2 else 0.0
        bos_s = f"{vb[-1]*100:4.1f}%" if vb else "  -  "
        vd_s = f"{vd[-1]:.4f}" if vd else "   -   "
        traj = f"{vm[0]:.4f}->{vm[mid]:.4f}->{vm[-1]:.4f}"
        print(f"  {hf.parent.name[:46]:46s} | {steps[0]:5d}->{steps[-1]:<6d} | {traj:26s} | {tail_delta:+.4f} | {vd_s:7s} | {bos_s}")
        out[hf.parent.name] = {"steps": steps, "val_mel": vm, "val_dur": vd, "val_bos_ratio": vb, "tail_delta": round(tail_delta, 4)}
    return out


def make_batch(entries, manifest_dir: Path, stats: dict, n_mels: int, device: torch.device):
    mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)[:, None]
    mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)[:, None]
    items = []
    for e in entries:
        d = torch.load(resolve_feat_path(e, manifest_dir), map_location="cpu", weights_only=False)
        items.append({
            "phoneme_ids": d["phoneme_ids"],
            "lang_ids": d["lang_ids"],
            "mel": (d["mel"].float() - mel_mean) / mel_std,
            "mel_raw": d["mel"].float(),
            "f0": (d["log_f0"].float() - stats["f0_mean"]) / stats["f0_std"],
            "energy": (d["log_energy"].float() - stats["e_mean"]) / stats["e_std"],
            "wav": d["wav"],
            "text": d.get("text", e.get("text", "")),
            "utt_id": d.get("utt_id", e.get("utt_id", "")),
        })
    B = len(items)
    T_max = max(it["phoneme_ids"].shape[0] for it in items)
    F_max = max(it["mel"].shape[-1] for it in items)
    pids = torch.zeros(B, T_max, dtype=torch.long)
    lids = torch.zeros(B, T_max, dtype=torch.long)
    t_lens = torch.zeros(B, dtype=torch.long)
    mels = torch.zeros(B, n_mels, F_max, dtype=torch.float32)
    f0s = torch.zeros(B, F_max, dtype=torch.float32)
    es = torch.zeros(B, F_max, dtype=torch.float32)
    m_lens = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        t, f = it["phoneme_ids"].shape[0], it["mel"].shape[-1]
        pids[i, :t] = it["phoneme_ids"]; lids[i, :t] = it["lang_ids"]; t_lens[i] = t
        mels[i, :, :f] = it["mel"]; f0s[i, :f] = it["f0"]; es[i, :f] = it["energy"]; m_lens[i] = f
    return (
        pids.to(device), lids.to(device), t_lens.to(device),
        mels.to(device), f0s.to(device), es.to(device), m_lens.to(device),
        items,
    )


@torch.no_grad()
def section_b_train_val_loss(model, meta: dict, manifest_dir: Path, stats: dict, n_mels: int, device: torch.device):
    print("\n" + "=" * 78)
    print("[B] Train vs Val 分项损失分解（含归一化 vs 原始 log-mel L1 与估算 dB 误差）")
    print("=" * 78)
    mel_std = np.asarray(stats["mel_std"], dtype=np.float32)
    mel_std_t = torch.tensor(mel_std, device=device, dtype=torch.float32)[None, :, None]
    print(f"  MEL_STD 统计: min={mel_std.min():.3f}, mean={mel_std.mean():.3f}, max={mel_std.max():.3f}")
    model.eval()
    res = {}
    for split in ("train", "val"):
        entries = meta[split][:64]
        acc = {k: 0.0 for k in ("mel", "mel_raw", "dur", "f0", "energy", "align", "bin", "bos_ratio")}
        n_b = 0
        for st in range(0, len(entries), 16):
            sub = entries[st : st + 16]
            if not sub:
                continue
            pids, lids, tl, mels, f0s, es, ml, _ = make_batch(sub, manifest_dir, stats, n_mels, device)
            out = model.forward_train(pids, lids, tl, mels, f0s, es, ml, step_ratio=1.0)
            # 同时精确计算原始 log-mel L1（乘以逐 bin mel_std）
            B, T_max = pids.shape
            F_max = mels.shape[-1]
            tmask = torch.arange(T_max, device=device)[None, :] < tl[:, None]
            mmask = torch.arange(F_max, device=device)[None, :] < ml[:, None]
            x_emb = model.embed_text(pids, lids, tmask)
            h_txt = model.backbone(x_emb, mask=tmask)
            prior = model._diag_build_prior(tl, ml, T_max, F_max)
            alp, _ = model.aligner(mels, x_emb, tmask, attn_log_prior=prior)
            ah, dur_gt = model._diag_viterbi(alp, tl, ml)
            h_frame = (torch.bmm(ah.to(h_txt.dtype), h_txt) + model._frame_phase_embed(dur_gt, F_max, h_txt.dtype)) * mmask[:, :, None].to(h_txt.dtype)
            f0_p = model.f0_predictor(h_frame)
            e_p = model.energy_predictor(h_frame)
            h_dec = (h_frame + model.f0_emb(f0_p.unsqueeze(-1)) + model.energy_emb(e_p.unsqueeze(-1))) * mmask[:, :, None].to(h_frame.dtype)
            for blk in model.frame_decoder:
                h_dec = blk(h_dec)
            mel_pred = model.mel_out(h_dec).transpose(1, 2)
            raw_l1 = ((mel_pred - mels).abs() * mel_std_t * mmask[:, None, :]).sum() / (mmask.sum().clamp(min=1) * n_mels)

            for k in ("mel", "dur", "f0", "energy", "align", "bin", "bos_ratio"):
                acc[k] += float(out[k].item())
            acc["mel_raw"] += float(raw_l1.item())
            n_b += 1
        for k in acc:
            acc[k] /= max(n_b, 1)
        db_err = acc["mel_raw"] * 8.6858896
        print(
            f"  [{split:5s}] mel_norm={acc['mel']:.4f} | mel_raw={acc['mel_raw']:.4f} (~{db_err:.1f} dB) | "
            f"dur={acc['dur']:.4f} | f0={acc['f0']:.4f} | eng={acc['energy']:.4f} | "
            f"align={acc['align']:.3f} | bin={acc['bin']:.4f} | bos={acc['bos_ratio']*100:.1f}%"
        )
        res[split] = acc
    gap = res["val"]["mel"] - res["train"]["mel"]
    print(f"  -> Val - Train mel_norm 差值: {gap:+.4f} ({'轻微/正常泛化差' if gap < 0.08 else '明显过拟合倾向'})")
    return res


@torch.no_grad()
def section_c_aligner_forensics(model, frontend, meta: dict, manifest_dir: Path, stats: dict, n_mels: int, device: torch.device, out_dir: Path, num_utts: int = 5):
    print("\n" + "=" * 78)
    print("[C] Aligner 深度体检（锐度 / 对角偏离 / dur==1 塌缩率 / dur_gt vs dur_pred）")
    print("=" * 78)
    model.eval()
    summary = {}
    for split in ("train", "val"):
        entries = meta[split][:num_utts]
        soft_max_ps, eff_toks_l, diag_devs, bos_rs, eos_rs, pct_d1_all, pct_d1_mid = [], [], [], [], [], [], []
        dur_maes, dur_rs, len_ratios, pct_p1_all = [], [], [], []
        for u_idx, e in enumerate(entries):
            pids, lids, tl, mels, f0s, es, ml, items = make_batch([e], manifest_dir, stats, n_mels, device)
            T_b, F_b = int(tl[0].item()), int(ml[0].item())
            tmask = torch.ones(1, T_b, device=device, dtype=torch.bool)
            x_emb = model.embed_text(pids, lids, tmask)
            h_txt = model.backbone(x_emb, mask=tmask)
            prior = model._diag_build_prior(tl, ml, T_b, F_b)
            alp, asoft = model.aligner(mels, x_emb, tmask, attn_log_prior=prior)
            ahard, dur_gt_t = model._diag_viterbi(alp, tl, ml)

            log_dur_p = model.dur_predictor(h_txt)
            dur_pred_t = torch.clamp(torch.round(torch.exp(log_dur_p) - 1.0), min=1, max=60).long()

            asoft_np = asoft[0].cpu().numpy()   # (F, T)
            ahard_np = ahard[0].cpu().numpy()   # (F, T)
            dur_gt = dur_gt_t[0].cpu().numpy()
            dur_pr = dur_pred_t[0].cpu().numpy()

            max_p = asoft_np.max(axis=-1)
            ent = -(asoft_np * np.log(np.clip(asoft_np, 1e-12, 1.0))).sum(axis=-1)
            t_path = ahard_np.argmax(axis=-1)
            diag_t = np.linspace(0, max(T_b - 1, 0), F_b)

            soft_max_ps.append(float(max_p.mean()))
            eff_toks_l.append(float(np.exp(ent).mean()))
            diag_devs.append(float(np.abs(t_path - diag_t).mean()))
            bos_rs.append(float(dur_gt[0] / max(F_b, 1)))
            eos_rs.append(float(dur_gt[-1] / max(F_b, 1)))
            pct_d1_all.append(float((dur_gt == 1).mean()))
            pct_d1_mid.append(float((dur_gt[1:-1] == 1).mean()) if T_b > 2 else 0.0)
            dur_maes.append(float(np.abs(dur_pr - dur_gt).mean()))
            dur_rs.append(pearson_r(dur_gt.astype(np.float64), dur_pr.astype(np.float64)))
            len_ratios.append(float(dur_pr.sum() / max(dur_gt.sum(), 1)))
            pct_p1_all.append(float((dur_pr == 1).mean()))

            if split == "val" and u_idx == 0:
                print(f"  示例 Val 句 [{items[0]['utt_id']}] T={T_b}, F={F_b} | 文本: {items[0]['text']}")
                print(f"  {'idx':>3s} | {'token':10s} | {'dur_gt':>6s} | {'dur_pr':>6s} | {'diff':>5s} | {'soft_maxP':>9s} | {'帧区间':>11s}")
                f_cur = 0
                for ti in range(min(T_b, 32)):
                    dg, dp_i = int(dur_gt[ti]), int(dur_pr[ti])
                    tok_s = frontend.id2tok.get(int(pids[0, ti].item()), "?")
                    seg_p = float(asoft_np[f_cur : f_cur + dg, ti].mean()) if dg > 0 else 0.0
                    print(f"  {ti:3d} | {tok_s:10s} | {dg:6d} | {dp_i:6d} | {dp_i-dg:+5d} | {seg_p:9.3f} | {f_cur:4d}..{f_cur+dg-1:<4d}")
                    f_cur += dg
                if T_b > 32:
                    print(f"  ... (省略后 {T_b - 32} 个 token)")

            if split == "val" and u_idx < 3:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.imshow(asoft_np.T, origin="lower", aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
                    ax.plot(np.arange(F_b), t_path, color="red", linewidth=1.2, label="Viterbi hard")
                    ax.set_title(f"{items[0]['utt_id']} | maxP={max_p.mean():.2f}, effTok={np.exp(ent).mean():.2f}, dur_r={dur_rs[-1]:.2f}")
                    ax.set_xlabel("Mel Frame"); ax.set_ylabel("Phoneme Index"); ax.legend(loc="upper left")
                    plt.tight_layout()
                    fig.savefig(out_dir / f"align_{items[0]['utt_id']}.png", dpi=130)
                    plt.close(fig)
                except Exception:
                    pass

        s = {
            "soft_max_p": round(float(np.mean(soft_max_ps)), 3),
            "eff_tokens": round(float(np.mean(eff_toks_l)), 2),
            "diag_dev_toks": round(float(np.mean(diag_devs)), 2),
            "bos_pct": round(float(np.mean(bos_rs) * 100), 1),
            "eos_pct": round(float(np.mean(eos_rs) * 100), 1),
            "gt_dur1_pct": round(float(np.mean(pct_d1_mid) * 100), 1),
            "pred_dur1_pct": round(float(np.mean(pct_p1_all) * 100), 1),
            "dur_mae_frames": round(float(np.mean(dur_maes)), 2),
            "dur_pearson_r": round(float(np.nanmean(dur_rs)), 3),
            "len_ratio": round(float(np.mean(len_ratios)), 3),
        }
        summary[split] = s
        print(
            f"  [{split:5s}] soft_maxP={s['soft_max_p']:.3f} | effTok={s['eff_tokens']:.2f} | "
            f"diagDev={s['diag_dev_toks']:.2f} | bos={s['bos_pct']:.1f}% eos={s['eos_pct']:.1f}% | "
            f"gt_dur1={s['gt_dur1_pct']:.1f}% | durMAE={s['dur_mae_frames']:.2f}f | r={s['dur_pearson_r']:.3f} | lenRatio={s['len_ratio']:.3f}"
        )
    print(f"  [Saved] 对齐热力图 PNG 已保存到: {out_dir}/align_*.png")
    return summary


def section_d_frontend_audit(frontend, meta: dict, manifest_dir: Path, max_utts: int = 30):
    print("\n" + "=" * 78)
    print("[D] 前端一致性审计（训练缓存 BZNSYP prosody_pinyin vs 推理 pypinyin）")
    print("=" * 78)
    entries = meta["val"][:max_utts]
    exact, tot_cached, tot_infer, unk_cached, unk_infer, tot_ed = 0, 0, 0, 0, 0, 0
    examples = []
    for e in entries:
        d = torch.load(resolve_feat_path(e, manifest_dir), map_location="cpu", weights_only=False)
        cached_ids = d["phoneme_ids"].tolist()
        text = d.get("text", e.get("text", ""))
        infer_ids, _ = frontend.text_to_ids(text)
        if cached_ids == infer_ids:
            exact += 1
        ed = edit_distance(cached_ids, infer_ids)
        tot_ed += ed
        tot_cached += len(cached_ids)
        tot_infer += len(infer_ids)
        unk_cached += sum(1 for x in cached_ids if x == frontend.unk_id)
        unk_infer += sum(1 for x in infer_ids if x == frontend.unk_id)
        if ed > 0 and len(examples) < 3:
            c_toks = [frontend.id2tok.get(x, "?") for x in cached_ids]
            i_toks = [frontend.id2tok.get(x, "?") for x in infer_ids]
            examples.append((d.get("utt_id", e.get("utt_id", "")), text, ed, len(cached_ids), c_toks, i_toks))
    n = max(len(entries), 1)
    mismatch_rate = tot_ed / max(tot_cached, 1) * 100.0
    print(
        f"  检查 {len(entries)} 句 Val: 完全一致 {exact}/{len(entries)} ({exact/n*100:.1f}%) | "
        f"音素编辑差异率 {mismatch_rate:.2f}% | "
        f"<unk> 率: cached={unk_cached/max(tot_cached,1)*100:.2f}%, infer={unk_infer/max(tot_infer,1)*100:.2f}%"
    )
    for uid, txt, ed, ln, ct, it in examples:
        print(f"  差异示例 [{uid}] (ed={ed}/{ln}): {txt}")
        print(f"    cached: {' '.join(ct[:24])}{' ...' if len(ct)>24 else ''}")
        print(f"    infer : {' '.join(it[:24])}{' ...' if len(it)>24 else ''}")
    return {"exact_rate": round(exact / n, 3), "phoneme_diff_pct": round(mismatch_rate, 2), "unk_cached": unk_cached, "unk_infer": unk_infer}


@torch.no_grad()
def section_e_link_isolation(
    model, frontend, causal_voc, pretrained_vocos, meta: dict, manifest_dir: Path, stats: dict,
    n_mels: int, sr: int, device: torch.device, out_dir: Path, num_utts: int = 5,
    run_utmos: bool = True, run_asr: bool = True, whisper_name: str = "small",
):
    print("\n" + "=" * 78)
    print(f"[E] 8 链路逐级隔离评测（{num_utts} 句 Val，含 Teacher-Forced Mel -> 预训练 Vocos）")
    print("=" * 78)
    import librosa

    utmos_model = None
    if run_utmos:
        try:
            utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        except Exception as e:
            print(f"  [Warn] UTMOS 加载失败: {e}")

    whisper_model = None
    if run_asr:
        try:
            from faster_whisper import WhisperModel
            # 强制走 CPU int8，彻底避开 ctranslate2 缺 libcublas.so.12 导致的全 NaN
            print(f"  [Eval] 加载 Faster-Whisper ({whisper_name}, CPU int8)...")
            whisper_model = WhisperModel(whisper_name, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"  [Warn] Faster-Whisper 加载失败: {e}")

    def _utmos(wav_np):
        if utmos_model is None or len(wav_np) < 800:
            return float("nan")
        w16 = librosa.resample(wav_np.astype(np.float32), orig_sr=sr, target_sr=16000)
        return float(utmos_model(torch.from_numpy(w16).unsqueeze(0).to(device), 16000).item())

    def _asr(wav_np, ref_text):
        if whisper_model is None or len(wav_np) < 800:
            return float("nan"), ""
        w16 = np.ascontiguousarray(librosa.resample(wav_np.astype(np.float32), orig_sr=sr, target_sr=16000), dtype=np.float32)
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", ref_text) else "en"
        segs, _ = whisper_model.transcribe(w16, language=lang, beam_size=1)
        hyp = "".join(s.text for s in segs).strip()
        return compute_cer(ref_text, hyp), hyp

    mel_mean = torch.tensor(stats["mel_mean"], device=device, dtype=torch.float32)[None, :, None]
    mel_std = torch.tensor(stats["mel_std"], device=device, dtype=torch.float32)[None, :, None]
    def denorm(m): return m * mel_std + mel_mean

    def decode_pre(m_raw):
        if pretrained_vocos is None:
            return None
        return np.clip(pretrained_vocos.decode(m_raw).squeeze(0).cpu().float().numpy(), -1.0, 1.0)

    def decode_causal(m_raw):
        if causal_voc is None:
            return None
        return np.clip(causal_voc(m_raw).squeeze(0).cpu().float().numpy(), -1.0, 1.0)

    links = [
        ("0_gt_wav",               "GT 真声"),
        ("1_gt_mel_prevocos",      "GT mel -> PreVocos"),
        ("2_gt_mel_causalvoc",     "GT mel -> CausalVoc"),
        ("3_tf_gt_f0_prevocos",    "TF(dur_gt+f0_gt+e_gt) -> PreVocos"),
        ("4_tf_pred_f0_prevocos",  "TF(dur_gt+f0_pr+e_pr) -> PreVocos"),
        ("5_inf_cached_prevocos",  "Infer(cached_pids+dur_pr) -> PreVocos"),
        ("6_inf_pypinyin_prevocos","Infer(pypinyin+dur_pr) -> PreVocos"),
        ("7_inf_pypinyin_causal",  "Infer(pypinyin+dur_pr) -> CausalVoc"),
    ]
    scores = {k: {"utmos": [], "cer": [], "norm_l1": [], "raw_l1": [], "dtw_raw_l1": []} for k, _ in links}
    first_transcripts = {}
    wav_root = out_dir / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    model.eval()

    for u_idx, e in enumerate(meta["val"][:num_utts]):
        pids, lids, tl, mels_norm, f0s, es, ml, items = make_batch([e], manifest_dir, stats, n_mels, device)
        it = items[0]
        uid, text = it["utt_id"], it["text"]
        T_b, F_b = int(tl[0].item()), int(ml[0].item())
        gt_wav = (it["wav"].float() / 32767.0).numpy() if it["wav"].dtype == torch.int16 else it["wav"].float().numpy()
        gt_mel_raw = it["mel_raw"].unsqueeze(0).to(device)  # (1, 100, F)
        gt_mel_raw_np = it["mel_raw"].numpy()

        tmask = torch.ones(1, T_b, device=device, dtype=torch.bool)
        x_emb = model.embed_text(pids, lids, tmask)
        h_txt = model.backbone(x_emb, mask=tmask)
        prior = model._diag_build_prior(tl, ml, T_b, F_b)
        alp, _ = model.aligner(mels_norm, x_emb, tmask, attn_log_prior=prior)
        ahard, dur_gt = model._diag_viterbi(alp, tl, ml)

        # Link 3: TF with dur_gt + f0_gt + e_gt
        h_frame = torch.bmm(ahard.to(h_txt.dtype), h_txt) + model._frame_phase_embed(dur_gt, F_b, h_txt.dtype)
        h_dec3 = h_frame + model.f0_emb(f0s.unsqueeze(-1)) + model.energy_emb(es.unsqueeze(-1))
        for blk in model.frame_decoder:
            h_dec3 = blk(h_dec3)
        mel3_norm = model.mel_out(h_dec3).transpose(1, 2)
        mel3_raw = denorm(mel3_norm)

        # Link 4: TF with dur_gt + f0_pred + e_pred (精确对应 val_mel 路径)
        f0_p4, e_p4 = model.f0_predictor(h_frame), model.energy_predictor(h_frame)
        h_dec4 = h_frame + model.f0_emb(f0_p4.unsqueeze(-1)) + model.energy_emb(e_p4.unsqueeze(-1))
        for blk in model.frame_decoder:
            h_dec4 = blk(h_dec4)
        mel4_norm = model.mel_out(h_dec4).transpose(1, 2)
        mel4_raw = denorm(mel4_norm)

        # Link 5: Infer with cached prosody_pinyin pids
        mel5_norm, _, _, _ = model.infer_offline(pids, lids)
        mel5_raw = denorm(mel5_norm)

        # Link 6 & 7: Infer with pypinyin text_to_ids(text)
        p_py, l_py = frontend.text_to_ids(text)
        mel6_norm, _, _, _ = model.infer_offline(
            torch.tensor([p_py], dtype=torch.long, device=device),
            torch.tensor([l_py], dtype=torch.long, device=device),
        )
        mel6_raw = denorm(mel6_norm)

        link_wavs = {
            "0_gt_wav":                (gt_wav, None, None),
            "1_gt_mel_prevocos":       (decode_pre(gt_mel_raw), None, None),
            "2_gt_mel_causalvoc":      (decode_causal(gt_mel_raw), None, None),
            "3_tf_gt_f0_prevocos":     (decode_pre(mel3_raw), mel3_norm, mel3_raw),
            "4_tf_pred_f0_prevocos":   (decode_pre(mel4_raw), mel4_norm, mel4_raw),
            "5_inf_cached_prevocos":   (decode_pre(mel5_raw), mel5_norm, mel5_raw),
            "6_inf_pypinyin_prevocos": (decode_pre(mel6_raw), mel6_norm, mel6_raw),
            "7_inf_pypinyin_causal":   (decode_causal(mel6_raw), mel6_norm, mel6_raw),
        }

        u_dir = wav_root / uid
        u_dir.mkdir(parents=True, exist_ok=True)
        for k, (w, m_n, m_r) in link_wavs.items():
            if w is not None:
                sf.write(u_dir / f"{k}.wav", w, sr)
                scores[k]["utmos"].append(_utmos(w))
                cer, hyp = _asr(w, text)
                scores[k]["cer"].append(cer)
                if u_idx == 0:
                    first_transcripts[k] = (cer, hyp)
            if m_n is not None and m_n.shape[-1] == mels_norm.shape[-1]:
                scores[k]["norm_l1"].append(float(F.l1_loss(m_n, mels_norm).item()))
                scores[k]["raw_l1"].append(float(F.l1_loss(m_r, gt_mel_raw).item()))
            if m_r is not None:
                scores[k]["dtw_raw_l1"].append(dtw_mel_l1(m_r.squeeze(0).cpu().numpy(), gt_mel_raw_np))

    def _m(arr):
        v = [x for x in arr if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n  {'链路':38s} | {'UTMOS↑':>7s} | {'CER↓':>7s} | {'NormL1':>7s} | {'RawL1':>7s} | {'DTW_RawL1':>9s}")
    print("  " + "-" * 90)
    table = {}
    for k, label in links:
        u = _m(scores[k]["utmos"])
        c = _m(scores[k]["cer"])
        nl = _m(scores[k]["norm_l1"])
        rl = _m(scores[k]["raw_l1"])
        dl = _m(scores[k]["dtw_raw_l1"])
        fmt = lambda v, p=3: (f"{v:.{p}f}" if not math.isnan(v) else "   -   ")
        fmt_c = lambda v: (f"{v*100:6.1f}%" if not math.isnan(v) else "   -   ")
        print(f"  {label:38s} | {fmt(u):>7s} | {fmt_c(c):>7s} | {fmt(nl,4):>7s} | {fmt(rl,4):>7s} | {fmt(dl,4):>9s}")
        table[k] = {"utmos": u, "cer": c, "norm_l1": nl, "raw_l1": rl, "dtw_raw_l1": dl}

    if first_transcripts:
        ref0 = meta["val"][0]["text"]
        print(f"\n  [首句 Whisper 识别对照] GT 原文: {ref0}")
        for k, label in links:
            if k in first_transcripts:
                cer0, hyp0 = first_transcripts[k]
                c_s = f"{cer0*100:5.1f}%" if not math.isnan(cer0) else "  -  "
                print(f"    {label:38s} | CER={c_s} | 识别: {hyp0}")

    # 自动归因判词
    print("\n  [自动诊断判词]")
    u1 = table["1_gt_mel_prevocos"]["utmos"]
    u2 = table["2_gt_mel_causalvoc"]["utmos"]
    u3 = table["3_tf_gt_f0_prevocos"]["utmos"]
    u4 = table["4_tf_pred_f0_prevocos"]["utmos"]
    u5 = table["5_inf_cached_prevocos"]["utmos"]
    u6 = table["6_inf_pypinyin_prevocos"]["utmos"]
    c3 = table["3_tf_gt_f0_prevocos"]["cer"]
    c5 = table["5_inf_cached_prevocos"]["cer"]
    c3_s = f"{c3*100:.1f}%" if not math.isnan(c3) else "N/A"
    c5_s = f"{c5*100:.1f}%" if not math.isnan(c5) else "N/A"
    if not math.isnan(u1) and not math.isnan(u2) and (u1 - u2) > 0.8:
        print(f"  1) 声码器失真显著: GT mel 经 PreVocos ({u1:.2f}) -> CausalVoc ({u2:.2f}), Δ={u1-u2:.2f}（电音主因：旧版 GAN/FM 权重被 /7 和 /35 缩水且 GAN 仅 6k 步）。")
    if not math.isnan(u3) and (u3 < 2.4 or (not math.isnan(c3) and c3 > 0.25)):
        print(f"  2) ★ Teacher-Forced 最强上界 (dur_gt + f0_gt + e_gt) 本身已经听不清 (UTMOS={u3:.2f}, CER={c3_s})！")
        print("     -> 根因锁定在【Aligner 对齐发糊/锁歪】或【Backbone+frame_decoder 欠容量/纯L1过平滑】，而非单纯 dur_predictor！")
    elif not math.isnan(u3) and not math.isnan(u4) and (u3 - u4) > 0.4:
        print(f"  2) ★ F0/Energy 预测器拖累明显: TF(f0_gt)={u3:.2f} -> TF(f0_pred)={u4:.2f}, Δ={u3-u4:.2f}。")
    elif not math.isnan(u4) and not math.isnan(u5) and ((u4 - u5) > 0.5 or (not math.isnan(c5) and not math.isnan(c3) and (c5 - c3) > 0.2)):
        print(f"  2) ★ TF 清晰但预测时长推理大滑坡: TF(f0_pred)={u4:.2f} -> Infer(cached)={u5:.2f} (CER {c3_s} -> {c5_s})！")
        print("     -> 根因锁定在【dur_predictor 预测不准 / 对齐时长噪声过大】。")
    if not math.isnan(u5) and not math.isnan(u6) and abs(u5 - u6) > 0.25:
        print(f"  3) 前端差异影响: cached_pids ({u5:.2f}) vs pypinyin ({u6:.2f})，需对齐训练/推理音素前端。")
    print(f"  [Saved] 8 链路试听 WAV 已保存到: {wav_root}")
    return table


def section_f_chunk_attn_benchmark(model, device: torch.device, sr: int = 24000, hop_length: int = 256):
    print("\n" + "=" * 78)
    print("[F] ChunkLookaheadBackbone 零循环并行 等价性验证 & GPU/CPU 提速实测")
    print("=" * 78)
    if model.backbone_type != "chunk_attn":
        print(f"  (当前模型为 {model.backbone_type}，跳过 chunk_attn 专项压测)")
        return {}
    bb = model.backbone
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # 1) 精确 FP32 前反向数值等价验证（临时关闭 TF32 避免 Ampere/Ada FP32 舍入噪声干扰门禁）
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    bb_f32 = bb.float().train()
    torch.manual_seed(123)
    T_lens = [75, 43, 19, 3]
    B_t, T_t, D = len(T_lens), max(T_lens), model.d_model
    mask_t = (torch.arange(T_t, device=device)[None, :] < torch.tensor(T_lens, device=device)[:, None])
    x_base = torch.randn(B_t, T_t, D, device=device) * mask_t[:, :, None].float()

    x_par = x_base.clone().requires_grad_(True)
    y_par = bb_f32.forward(x_par, mask=mask_t)
    (y_par.pow(2) * mask_t[:, :, None].float()).sum().backward()
    g_x_par = x_par.grad.clone()
    g_p_par = {n: p.grad.clone() for n, p in bb_f32.named_parameters()}

    bb_f32.zero_grad(set_to_none=True)
    x_seq = x_base.clone().requires_grad_(True)
    y_seq = bb_f32._forward_sequential(x_seq, mask=mask_t)
    (y_seq.pow(2) * mask_t[:, :, None].float()).sum().backward()

    fwd_diff = ((y_par.detach() - y_seq.detach()) * mask_t[:, :, None].float()).abs().max().item()
    xgrad_diff = ((g_x_par - x_seq.grad) * mask_t[:, :, None].float()).abs().max().item()
    pgrad_diff = max((g_p_par[n] - p.grad).abs().max().item() for n, p in bb_f32.named_parameters())
    bb_f32.zero_grad(set_to_none=True)

    # 流式 forward_step 等价 + 零泄漏检查
    bb_f32.eval()
    with torch.no_grad():
        xs = torch.randn(1, 75, D, device=device)
        y_off = bb_f32.forward(xs)
        st = bb_f32.init_state(1, device, xs.dtype)
        outs = []
        for s in range(0, 75, 4):
            o, st = bb_f32.forward_step(xs[:, s : s + 4], st, is_final=(s + 4 >= 75))
            if o.shape[1] > 0:
                outs.append(o)
        stream_diff = (y_off - torch.cat(outs, dim=1)).abs().max().item()

        xs_pert = xs.clone()
        xs_pert[:, 12 + bb_f32.lookahead :] = torch.randn(1, 75 - 12 - bb_f32.lookahead, D, device=device)
        leak = (y_off[:, :12] - bb_f32.forward(xs_pert)[:, :12]).abs().max().item()

    print(
        f"  [数值等价] Fwd={fwd_diff:.2e} | xGrad={xgrad_diff:.2e} | pGrad={pgrad_diff:.2e} | "
        f"Stream={stream_diff:.2e} | Leak={leak:.1e}"
    )
    assert fwd_diff < 1e-4 and xgrad_diff < 1e-3 and pgrad_diff < 1e-3 and stream_diff < 1e-4 and leak == 0.0

    # 2) GPU/Device 速度与显存实测（开启 TF32 + cuDNN benchmark，对齐真实 4090 训练环境）
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    bb_f32.train()
    bench_res = {"fwd_diff": fwd_diff, "xgrad_diff": xgrad_diff, "pgrad_diff": pgrad_diff, "stream_diff": stream_diff, "leak": leak}
    for bs in (32, 64):
        T_b = 100
        xb = torch.randn(bs, T_b, D, device=device)
        tl_b = torch.randint(40, T_b + 1, (bs,), device=device)
        tl_b[0] = T_b
        mb = torch.arange(T_b, device=device)[None, :] < tl_b[:, None]

        def _time_fn(fn, runs=10):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    fn(xb, mask=mb).sum().backward()
                bb_f32.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    fn(xb, mask=mb).sum().backward()
                bb_f32.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / runs * 1000.0
            mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
            return ms, mem_mb

        ms_par, mem_par = _time_fn(bb_f32.forward, runs=10)
        ms_seq, mem_seq = _time_fn(bb_f32._forward_sequential, runs=5)
        speedup = ms_seq / max(ms_par, 1e-6)
        print(
            f"  [Backbone fwd+bwd B={bs:2d}, T={T_b}] 串行={ms_seq:6.1f} ms ({mem_seq:5.0f}MB) -> "
            f"零循环并行={ms_par:5.1f} ms ({mem_par:5.0f}MB) | ★ 提速 {speedup:.2f}x"
        )
        bench_res[f"B{bs}"] = {"seq_ms": round(ms_seq, 2), "par_ms": round(ms_par, 2), "speedup": round(speedup, 2), "par_vram_mb": round(mem_par, 1)}

    # 3) 2 线程 CPU 流式单句骨干延迟实测
    prev_t = torch.get_num_threads()
    torch.set_num_threads(2)
    bb_cpu = bb_f32.to("cpu").eval()
    x_cpu = torch.randn(1, 60, D)
    with torch.no_grad():
        for _ in range(2):
            st = bb_cpu.init_state(1, torch.device("cpu"), torch.float32)
            for s in range(0, 60, 4):
                _, st = bb_cpu.forward_step(x_cpu[:, s : s + 4], st, is_final=(s + 4 >= 60))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            st = bb_cpu.init_state(1, torch.device("cpu"), torch.float32)
            for s in range(0, 60, 4):
                _, st = bb_cpu.forward_step(x_cpu[:, s : s + 4], st, is_final=(s + 4 >= 60))
            ts.append((time.perf_counter() - t0) * 1000.0)
    med_cpu_ms = float(np.median(ts))
    print(f"  [CPU 2线程流式骨干 T=60 (~3.5s 语音)] 中位总耗时={med_cpu_ms:.1f} ms (单块 ~{med_cpu_ms/15:.2f} ms, 已含 RoPE 缓存 + 末层剪枝)")
    bench_res["cpu2t_backbone_ms_T60"] = round(med_cpu_ms, 2)
    bb_cpu.to(device)
    torch.set_num_threads(prev_t)
    return bench_res


def main():
    ap = argparse.ArgumentParser(description="AetherTTS Phase 1 深度诊断与 ChunkAttn 提速验证脚本")
    ap.add_argument("--root", default="./aethertts_exp", help="实验产物根目录（包含 runs/, vocoder/, cache/）")
    ap.add_argument("--profile", default="standard_phase1")
    ap.add_argument("--dataset", default="bznsyp")
    ap.add_argument("--model", default="E3_chunk_attn_C4_L8")
    ap.add_argument("--ckpt", default="", help="可选：手动指定声学模型 checkpoint 路径")
    ap.add_argument("--vocoder-ckpt", default="", help="可选：手动指定 CausalVocosLite final.pt 路径")
    ap.add_argument("--cache-dir", default="", help="可选：手动指定特征缓存目录或 manifest.json")
    ap.add_argument("--num-utts", type=int, default=5, help="用于隔离评测与对齐分析的 val 句数（默认 5）")
    ap.add_argument("--whisper-model", default="small", help="CPU int8 Whisper 模型规格（默认 small 速度快；可设 large-v3-turbo）")
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = root / "diag_out" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"AetherTTS Phase 1 Diagnostic | Device={device} | Root={root}")
    print("=" * 78)

    defs = load_phase1_defs(root)
    frontend = defs["FRONTEND"]
    AetherAcousticModel = defs["AetherAcousticModel"]
    CausalVocosLite = defs["CausalVocosLite"]

    ckpt_path = find_checkpoint(root, args.model, args.profile, args.dataset, args.ckpt)
    print(f"[Load] 声学权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    stats = ckpt["stats"]
    bb_type = ckpt.get("backbone_type", "chunk_attn")
    lookahead = int(ckpt.get("lookahead", 8))

    model = AetherAcousticModel(
        vocab_size=len(ckpt.get("vocab", frontend.vocab)),
        backbone_type=bb_type,
        d_model=int(cfg.get("d_model", 256)),
        n_mels=int(cfg.get("n_mels", 100)),
        k_left=int(cfg.get("k_left", 3)),
        lookahead=lookahead,
        swa_window=int(cfg.get("swa_window", 48)),
        chunk_size=int(cfg.get("chunk_attn_size", 4)),
        frame_lookahead=int(cfg.get("frame_lookahead", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    # 挂接 aligner 辅助函数便于分节复用
    model._diag_build_prior = defs["build_beta_binomial_log_prior"]
    model._diag_viterbi = defs["viterbi_monotonic_alignment"]
    print(f"  -> 加载成功 (strict=True)! step={ckpt.get('step')} | best_val_mel={ckpt.get('best_val_mel')} | backbone={bb_type}, L={lookahead}")

    manifest_path = find_manifest(root, args.dataset, cfg.get("max_train_hours"), args.cache_dir)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    print(f"[Load] 特征清单: {manifest_path} ({meta.get('total_hours')}h, train={meta.get('train_count')}, val={meta.get('val_count')})")

    causal_voc = None
    voc_ckpt = find_vocoder_ckpt(root, args.profile, args.dataset, args.vocoder_ckpt)
    if voc_ckpt is not None:
        causal_voc = CausalVocosLite(
            n_mels=int(cfg.get("n_mels", 100)),
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 256)),
            lookahead=int(cfg.get("vocoder_lookahead", 2)),
        ).to(device)
        causal_voc.load_state_dict(torch.load(voc_ckpt, map_location=device, weights_only=True), strict=True)
        causal_voc.eval()
        print(f"[Load] 自训声码器: {voc_ckpt}")
    else:
        print("[Warn] 未找到 CausalVocosLite final.pt，跳过自训声码器链路。")

    pretrained_vocos = None
    try:
        from vocos import Vocos
        pretrained_vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        print("[Load] 预训练 Vocos (charactr/vocos-mel-24khz) 就绪。")
    except Exception as e:
        print(f"[Warn] 预训练 Vocos 加载失败 ({e})。")

    report = {"ckpt": str(ckpt_path), "step": ckpt.get("step"), "best_val_mel": ckpt.get("best_val_mel")}
    report["history"] = section_a_history(root)
    report["loss_split"] = section_b_train_val_loss(model, meta, manifest_dir, stats, int(cfg.get("n_mels", 100)), device)
    report["aligner"] = section_c_aligner_forensics(model, frontend, meta, manifest_dir, stats, int(cfg.get("n_mels", 100)), device, out_dir, num_utts=args.num_utts)
    report["frontend"] = section_d_frontend_audit(frontend, meta, manifest_dir, max_utts=30)
    report["isolation"] = section_e_link_isolation(
        model, frontend, causal_voc, pretrained_vocos, meta, manifest_dir, stats,
        int(cfg.get("n_mels", 100)), int(cfg.get("sample_rate", 24000)), device, out_dir,
        num_utts=args.num_utts, run_utmos=(not args.skip_utmos), run_asr=(not args.skip_asr),
        whisper_name=args.whisper_model,
    )
    if not args.skip_bench:
        report["chunk_attn_bench"] = section_f_chunk_attn_benchmark(model, device, int(cfg.get("sample_rate", 24000)), int(cfg.get("hop_length", 256)))

    summary_path = out_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"[Done] 完整诊断 JSON 已保存至: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
