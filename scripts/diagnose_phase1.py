#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a9) 深度诊断与 ChunkAttn 提速校验脚本。

直接 import scripts/aethertts_lib.py（零正则 exec 脆弱耦合），复用 `./aethertts_exp` 产物，
一次性检查：
  A. 模型训练历史 (history.json) 收敛轨迹（val_mel / val_infer_mel / val_f0 / val_dur / best_step）
  B. Train vs Val 各分项损失分解（含归一化与原始 log-mel L1、共振峰结构损失、浊音 F0 与 UV 准确率）
  C. Aligner 深度体检（锐度、对角偏离、dur==1 塌缩率、dur_gt vs dur_pred 相关性 + 热力图 PNG）
  D. 前端一致性审计（v4 缓存标贝儿化/韵律音素 vs 推理端文本 G2P 差异与 0-<unk> 校验）
  E. 8 链路逐级隔离评测（含 Teacher-Forced + 真/预测 F0+UV -> PreVocos / CausalVoc+Harm + CER + UTMOS + DTW RawL1 + Mod_93Hz + Energy_Holes）
  F. ChunkLookaheadBackbone 零循环并行 & 局部相对 RoPE 超长流式等价校验

用法：
    python scripts/diagnose_phase1.py --root ./aethertts_exp
    python scripts/diagnose_phase1.py --root ./aethertts_exp --profile full_quality --num-utts 5
"""
import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import aethertts_lib as lib


def find_checkpoint(root: Path, model_name: str, profile: str, dataset: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        assert p.exists(), f"指定 checkpoint 不存在: {p}"
        return p
    runs_dir = root / "runs"
    cands = (
        sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_a10*"))
        or sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_a9*"))
        or sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_a8*"))
        or sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_a7*"))
        or sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_a6*"))
    ) if runs_dir.exists() else []
    if not cands and runs_dir.exists():
        cands = sorted(runs_dir.glob(f"{model_name}_{dataset}_{profile}_*")) or sorted(runs_dir.glob(f"{model_name}*"))
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
    cands = (
        sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_a10*"))
        or sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_a9*"))
        or sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_a8*"))
        or sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_a7*"))
        or sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_a6*"))
        or sorted(voc_dir.glob(f"causal_vocos_lite_{dataset}_{profile}_*"))
        or sorted(voc_dir.glob("causal_vocos_lite_*"))
    )
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
        for ver in ("v5", "v4", "v3", "v2"):
            exact = cache_dir / f"{dataset}_{ver}_max{max_hours}h" / "manifest.json"
            if exact.exists():
                return exact
    cands = (
        sorted(cache_dir.glob(f"{dataset}_v5_max*h/manifest.json"))
        or sorted(cache_dir.glob(f"{dataset}_v4_max*h/manifest.json"))
    ) if cache_dir.exists() else []
    if not cands and cache_dir.exists():
        cands = sorted(cache_dir.glob(f"{dataset}_*_max*h/manifest.json")) or sorted(cache_dir.glob("*/manifest.json"))
    if not cands:
        raise FileNotFoundError(f"在 {cache_dir} 中未找到 manifest.json，请通过 --cache-dir 指定。")
    return cands[-1]


def resolve_feat_path(entry: dict, manifest_dir: Path) -> Path:
    p = Path(entry["feat_path"])
    if p.exists():
        return p
    for cand_dir in (manifest_dir, manifest_dir.parent / f"{manifest_dir.name.split('_max')[0]}_pool"):
        alt = cand_dir / p.name
        if alt.exists():
            return alt
    raise FileNotFoundError(f"特征文件不存在: {p}")


def dtw_mel_l1(pred_mel: np.ndarray, gt_mel: np.ndarray) -> float:
    F1, F2 = pred_mel.shape[-1], gt_mel.shape[-1]
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
    print("[A] 训练历史收敛轨迹 (runs/*/history.json)")
    print("=" * 78)
    runs_dir = root / "runs"
    hfiles = sorted(runs_dir.glob("*/history.json")) if runs_dir.exists() else []
    if not hfiles:
        print("  (未找到 history.json)")
        return {}
    out = {}
    print(f"  {'Run 目录':46s} | {'步数范围':13s} | {'val_mel 轨迹 (首->中->末)':26s} | {'尾段Δ':7s} | {'val_f0':7s} | {'bos%':5s}")
    print("  " + "-" * 118)
    for hf in hfiles:
        try:
            h = json.loads(hf.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = h.get("steps", [])
        vm = h.get("val_mel", [])
        vf0 = h.get("val_f0", [])
        vb = h.get("val_bos_ratio", [])
        if not steps or not vm:
            continue
        mid = len(vm) // 2
        tail_delta = (vm[-1] - vm[max(0, len(vm) - 3)]) if len(vm) >= 2 else 0.0
        bos_s = f"{vb[-1]*100:4.1f}%" if vb else "  -  "
        vf0_s = f"{vf0[-1]:.4f}" if vf0 else "   -   "
        traj = f"{vm[0]:.4f}->{vm[mid]:.4f}->{vm[-1]:.4f}"
        print(f"  {hf.parent.name[:46]:46s} | {steps[0]:5d}->{steps[-1]:<6d} | {traj:26s} | {tail_delta:+.4f} | {vf0_s:7s} | {bos_s}")
        out[hf.parent.name] = {"steps": steps, "val_mel": vm, "val_f0": vf0, "val_bos_ratio": vb, "tail_delta": round(tail_delta, 4)}
    return out


def make_batch(entries, manifest_dir: Path, stats: dict, n_mels: int, device: torch.device):
    mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)[:, None]
    mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)[:, None]
    items = []
    for e in entries:
        d = torch.load(resolve_feat_path(e, manifest_dir), map_location="cpu", weights_only=False)
        pids = d["phoneme_ids"]
        bids = d["boundary_ids"] if "boundary_ids" in d else torch.zeros_like(pids)
        uv = d["uv"].float() if "uv" in d else torch.ones_like(d["log_f0"].float())
        items.append({
            "phoneme_ids": pids,
            "lang_ids": d["lang_ids"],
            "boundary_ids": bids,
            "mel": (d["mel"].float() - mel_mean) / mel_std,
            "mel_raw": d["mel"].float(),
            "f0": (d["log_f0"].float() - stats["f0_mean"]) / stats["f0_std"],
            "energy": (d["log_energy"].float() - stats["e_mean"]) / stats["e_std"],
            "uv": uv,
            "wav": d["wav"],
            "text": d.get("text", e.get("text", "")),
            "utt_id": d.get("utt_id", e.get("utt_id", "")),
        })
    B = len(items)
    T_max = max(it["phoneme_ids"].shape[0] for it in items)
    F_max = max(it["mel"].shape[-1] for it in items)
    pids = torch.zeros(B, T_max, dtype=torch.long)
    lids = torch.zeros(B, T_max, dtype=torch.long)
    bids = torch.zeros(B, T_max, dtype=torch.long)
    t_lens = torch.zeros(B, dtype=torch.long)
    mels = torch.zeros(B, n_mels, F_max, dtype=torch.float32)
    f0s = torch.zeros(B, F_max, dtype=torch.float32)
    es = torch.zeros(B, F_max, dtype=torch.float32)
    uvs = torch.zeros(B, F_max, dtype=torch.float32)
    m_lens = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        t, f = it["phoneme_ids"].shape[0], it["mel"].shape[-1]
        pids[i, :t] = it["phoneme_ids"]; lids[i, :t] = it["lang_ids"]; bids[i, :t] = it["boundary_ids"]; t_lens[i] = t
        mels[i, :, :f] = it["mel"]; f0s[i, :f] = it["f0"]; es[i, :f] = it["energy"]; uvs[i, :f] = it["uv"]; m_lens[i] = f
    return (
        pids.to(device), lids.to(device), bids.to(device), t_lens.to(device),
        mels.to(device), f0s.to(device), es.to(device), uvs.to(device), m_lens.to(device),
        items,
    )


@torch.no_grad()
def section_b_train_val_loss(model, meta: dict, manifest_dir: Path, stats: dict, n_mels: int, device: torch.device):
    print("\n" + "=" * 78)
    print("[B] Train vs Val 分项损失分解（含共振峰结构损失、浊音 F0、UV 与原始 log-mel dB 误差）")
    print("=" * 78)
    mel_std = np.asarray(stats["mel_std"], dtype=np.float32)
    mel_std_t = torch.tensor(mel_std, device=device, dtype=torch.float32)[None, :, None]
    model.eval()
    res = {}
    for split in ("train", "val"):
        entries = meta[split][:64]
        keys = ("mel", "mel_raw", "struct", "dur", "f0", "uv", "energy", "align", "bin", "bos_ratio")
        acc = {k: 0.0 for k in keys}
        n_b = 0
        for st in range(0, len(entries), 16):
            sub = entries[st : st + 16]
            if not sub:
                continue
            pids, lids, bids, tl, mels, f0s, es, uvs, ml, _ = make_batch(sub, manifest_dir, stats, n_mels, device)
            out = model.forward_train(pids, lids, tl, mels, f0s, es, ml, step_ratio=1.0, boundary_ids=bids, uv_gt=uvs)
            mel_gta = model.extract_gta_mel(pids, lids, tl, mels, ml, boundary_ids=bids)
            mmask = torch.arange(mels.shape[-1], device=device)[None, :] < ml[:, None]
            raw_l1 = ((mel_gta - mels).abs() * mel_std_t * mmask[:, None, :]).sum() / (mmask.sum().clamp(min=1) * n_mels)
            for k in out:
                if k in acc:
                    acc[k] += float(out[k].item())
            acc["mel_raw"] += float(raw_l1.item())
            n_b += 1
        for k in acc:
            acc[k] /= max(n_b, 1)
        db_err = acc["mel_raw"] * 8.6858896
        print(
            f"  [{split:5s}] mel_norm={acc['mel']:.4f} | strc={acc['struct']:.4f} | mel_raw={acc['mel_raw']:.4f} (~{db_err:.1f}dB) | "
            f"align={acc['align']:.3f} | dur={acc['dur']:.4f} | f0_v={acc['f0']:.4f} | uv={acc['uv']:.4f} | bos={acc['bos_ratio']*100:.1f}%"
        )
        res[split] = acc
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
        soft_max_ps, eff_toks_l, diag_devs, bos_rs, eos_rs, pct_d1_mid = [], [], [], [], [], []
        dur_maes, dur_rs, len_ratios = [], [], []
        for u_idx, e in enumerate(entries):
            pids, lids, bids, tl, mels, f0s, es, uvs, ml, items = make_batch([e], manifest_dir, stats, n_mels, device)
            T_b, F_b = int(tl[0].item()), int(ml[0].item())
            tmask = torch.ones(1, T_b, device=device, dtype=torch.bool)
            mmask = torch.ones(1, F_b, device=device, dtype=torch.bool)
            x_emb = model.embed_text(pids, lids, tmask, boundary_ids=bids)
            h_txt = model.backbone(x_emb, mask=tmask)
            prior = lib.build_beta_binomial_log_prior(tl, ml, T_b, F_b)
            alp, asoft = model.aligner(mels, x_emb, tmask, attn_log_prior=prior, mel_mask=mmask)
            ahard, dur_gt_t = lib.viterbi_monotonic_alignment(F.log_softmax(alp, dim=-1), tl, ml)
            dur_pred_t = torch.clamp(torch.round(torch.exp(model.dur_predictor(h_txt)) - 1.0), min=1, max=60).long()

            asoft_np = asoft[0].cpu().numpy()
            ahard_np = ahard[0].cpu().numpy()
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
            pct_d1_mid.append(float((dur_gt[1:-1] == 1).mean()) if T_b > 2 else 0.0)
            dur_maes.append(float(np.abs(dur_pr - dur_gt).mean()))
            dur_rs.append(pearson_r(dur_gt.astype(np.float64), dur_pr.astype(np.float64)))
            len_ratios.append(float(dur_pr.sum() / max(dur_gt.sum(), 1)))

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
            "dur_mae_frames": round(float(np.mean(dur_maes)), 2),
            "dur_pearson_r": round(float(np.nanmean(dur_rs)), 3),
            "len_ratio": round(float(np.mean(len_ratios)), 3),
        }
        summary[split] = s
        print(
            f"  [{split:5s}] soft_maxP={s['soft_max_p']:.3f} | effTok={s['eff_tokens']:.2f} | "
            f"diagDev={s['diag_dev_toks']:.2f} | bos={s['bos_pct']:.1f}% | gt_dur1={s['gt_dur1_pct']:.1f}% | "
            f"durMAE={s['dur_mae_frames']:.2f}f | r={s['dur_pearson_r']:.3f} | lenRatio={s['len_ratio']:.3f}"
        )
    return summary


def section_d_frontend_audit(frontend, meta: dict, manifest_dir: Path, max_utts: int = 30):
    print("\n" + "=" * 78)
    print("[D] 前端一致性审计（v4 缓存音素 vs 推理 G2P 差异与 <unk> 校验）")
    print("=" * 78)
    entries = meta["val"][:max_utts]
    exact, tot_cached, tot_infer, unk_cached, unk_infer, tot_ed = 0, 0, 0, 0, 0, 0
    for e in entries:
        d = torch.load(resolve_feat_path(e, manifest_dir), map_location="cpu", weights_only=False)
        cached_ids = d["phoneme_ids"].tolist()
        text = d.get("text", e.get("text", ""))
        infer_ids, _, _ = frontend.text_to_ids(text)
        if cached_ids == infer_ids:
            exact += 1
        tot_ed += edit_distance(cached_ids, infer_ids)
        tot_cached += len(cached_ids)
        tot_infer += len(infer_ids)
        unk_cached += sum(1 for x in cached_ids if x == frontend.unk_id)
        unk_infer += sum(1 for x in infer_ids if x == frontend.unk_id)
    n = max(len(entries), 1)
    mismatch_rate = tot_ed / max(tot_cached, 1) * 100.0
    print(
        f"  检查 {len(entries)} 句 Val: 完全一致 {exact}/{len(entries)} ({exact/n*100:.1f}%) | "
        f"音素编辑差异率 {mismatch_rate:.2f}% | <unk>: cached={unk_cached}, infer={unk_infer}"
    )
    return {"exact_rate": round(exact / n, 3), "phoneme_diff_pct": round(mismatch_rate, 2), "unk_cached": unk_cached, "unk_infer": unk_infer}


@torch.no_grad()
def section_e_link_isolation(
    model, frontend, causal_voc, pretrained_vocos, meta: dict, manifest_dir: Path, stats: dict,
    n_mels: int, sr: int, device: torch.device, out_dir: Path, num_utts: int = 5,
    run_utmos: bool = True, run_asr: bool = True, whisper_name: str = "small",
):
    print("\n" + "=" * 78)
    print(f"[E] 8 链路逐级隔离评测（{num_utts} 句 Val，含 Teacher-Forced/GTA Mel -> PreVocos & CausalVoc+Harm）")
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

    def _mod93(wav_np):
        env = np.abs(np.asarray(wav_np, dtype=np.float64))
        dc = float(env.sum())
        if dc < 1e-8 or env.size < 1024:
            return float("nan")
        t = np.arange(env.size, dtype=np.float64)
        ang = (-2.0 * math.pi * (float(sr) / 256.0) / float(sr)) * t
        return float(math.hypot(float(np.dot(env, np.cos(ang))), float(np.dot(env, np.sin(ang)))) / dc)

    def _holes(wav_np):
        w = np.asarray(wav_np, dtype=np.float64)
        nf = w.size // 256
        if nf < 5:
            return 0
        rms = np.sqrt(np.mean(w[: nf * 256].reshape(nf, 256) ** 2, axis=1) + 1e-12)
        return int(sum(
            1 for f in range(2, nf - 2)
            if min(max(rms[f - 2], rms[f - 1]), max(rms[f + 1], rms[f + 2])) > 0.05
            and rms[f] < 0.35 * min(max(rms[f - 2], rms[f - 1]), max(rms[f + 1], rms[f + 2]))
        ))

    mel_mean = torch.tensor(stats["mel_mean"], device=device, dtype=torch.float32)[None, :, None]
    mel_std = torch.tensor(stats["mel_std"], device=device, dtype=torch.float32)[None, :, None]
    def denorm(m): return m * mel_std + mel_mean
    def decode_pre(m_raw): return None if pretrained_vocos is None else np.clip(pretrained_vocos.decode(m_raw).squeeze(0).cpu().float().numpy(), -1.0, 1.0)
    def decode_causal(m_raw): return None if causal_voc is None else np.clip(causal_voc(m_raw).squeeze(0).cpu().float().numpy(), -1.0, 1.0)

    links = [
        ("0_gt_wav",               "GT 真声"),
        ("1_gt_mel_prevocos",      "GT mel -> PreVocos"),
        ("2_gt_mel_causalvoc",     "GT mel -> CausalVoc+Harm"),
        ("3_tf_gt_f0_prevocos",    "TF(dur_gt+f0_gt+uv_gt) -> PreVocos"),
        ("4_tf_gta_prevocos",      "GTA(dur_gt+f0_pr+uv_pr) -> PreVocos"),
        ("5_inf_cached_prevocos",  "Infer(cached_pids) -> PreVocos"),
        ("6_inf_text_prevocos",    "Infer(text_g2p) -> PreVocos"),
        ("7_inf_text_causal",      "Infer(text_g2p) -> CausalVoc+Harm"),
    ]
    scores = {k: {"utmos": [], "cer": [], "norm_l1": [], "raw_l1": [], "dtw_raw_l1": [], "mod93": [], "holes": []} for k, _ in links}
    wav_root = out_dir / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    model.eval()

    for e in meta["val"][:num_utts]:
        pids, lids, bids, tl, mels_norm, f0s, es, uvs, ml, items = make_batch([e], manifest_dir, stats, n_mels, device)
        it = items[0]
        uid, text = it["utt_id"], it["text"]
        T_b, F_b = int(tl[0].item()), int(ml[0].item())
        gt_wav = lib.decode_wav_tensor(it["wav"]).numpy()
        gt_mel_raw = it["mel_raw"].unsqueeze(0).to(device)
        gt_mel_raw_np = it["mel_raw"].numpy()

        tmask = torch.ones(1, T_b, device=device, dtype=torch.bool)
        mmask = torch.ones(1, F_b, device=device, dtype=torch.bool)
        x_emb = model.embed_text(pids, lids, tmask, boundary_ids=bids)
        h_txt = model.backbone(x_emb, mask=tmask)
        prior = lib.build_beta_binomial_log_prior(tl, ml, T_b, F_b)
        alp, _ = model.aligner(mels_norm, x_emb, tmask, attn_log_prior=prior, mel_mask=mmask)
        ahard, dur_gt = lib.viterbi_monotonic_alignment(F.log_softmax(alp, dim=-1), tl, ml)

        h_frame = torch.bmm(ahard.to(h_txt.dtype), h_txt) + model._frame_phase_embed(dur_gt, F_b, h_txt.dtype)
        h_pre = model.frame_pre(h_frame) if hasattr(model, "frame_pre") else h_frame
        mel3_norm = model._condition_decoder(h_pre, f0s, uvs, es)
        mel3_raw = denorm(mel3_norm)

        mel4_norm = model.extract_gta_mel(pids, lids, tl, mels_norm, ml, boundary_ids=bids)
        mel4_raw = denorm(mel4_norm)

        mel5_norm, _, _, _ = model.infer_offline(pids, lids, boundary_ids=bids)
        mel5_raw = denorm(mel5_norm)

        p_py, l_py, b_py = frontend.text_to_ids(text)
        mel6_norm, _, _, _ = model.infer_offline(
            torch.tensor([p_py], dtype=torch.long, device=device),
            torch.tensor([l_py], dtype=torch.long, device=device),
            boundary_ids=torch.tensor([b_py], dtype=torch.long, device=device),
        )
        mel6_raw = denorm(mel6_norm)

        link_wavs = {
            "0_gt_wav":               (gt_wav, None, None),
            "1_gt_mel_prevocos":      (decode_pre(gt_mel_raw), None, None),
            "2_gt_mel_causalvoc":     (decode_causal(gt_mel_raw), None, None),
            "3_tf_gt_f0_prevocos":    (decode_pre(mel3_raw), mel3_norm, mel3_raw),
            "4_tf_gta_prevocos":      (decode_pre(mel4_raw), mel4_norm, mel4_raw),
            "5_inf_cached_prevocos":  (decode_pre(mel5_raw), mel5_norm, mel5_raw),
            "6_inf_text_prevocos":    (decode_pre(mel6_raw), mel6_norm, mel6_raw),
            "7_inf_text_causal":      (decode_causal(mel6_raw), mel6_norm, mel6_raw),
        }

        u_dir = wav_root / uid
        u_dir.mkdir(parents=True, exist_ok=True)
        for k, (w, m_n, m_r) in link_wavs.items():
            if w is not None:
                sf.write(u_dir / f"{k}.wav", w, sr)
                scores[k]["utmos"].append(_utmos(w))
                cer, _ = _asr(w, text)
                scores[k]["cer"].append(cer)
                scores[k]["mod93"].append(_mod93(w))
                scores[k]["holes"].append(_holes(w))
            if m_n is not None and m_n.shape[-1] == mels_norm.shape[-1]:
                scores[k]["norm_l1"].append(float(F.l1_loss(m_n, mels_norm).item()))
                scores[k]["raw_l1"].append(float(F.l1_loss(m_r, gt_mel_raw).item()))
            if m_r is not None:
                scores[k]["dtw_raw_l1"].append(dtw_mel_l1(m_r.squeeze(0).cpu().numpy(), gt_mel_raw_np))

    def _m(arr):
        v = [x for x in arr if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n  {'链路':38s} | {'UTMOS↑':>7s} | {'CER↓':>7s} | {'NormL1':>7s} | {'RawL1':>7s} | {'DTW_RawL1':>9s} | {'Mod93↓':>7s} | {'Holes↓':>6s}")
    print("  " + "-" * 108)
    table = {}
    for k, label in links:
        u, c, nl, rl, dl, m93 = (_m(scores[k][m]) for m in ("utmos", "cer", "norm_l1", "raw_l1", "dtw_raw_l1", "mod93"))
        hl = int(sum(scores[k]["holes"]))
        fmt = lambda v, p=3: (f"{v:.{p}f}" if not math.isnan(v) else "   -   ")
        fmt_c = lambda v: (f"{v*100:6.1f}%" if not math.isnan(v) else "   -   ")
        print(f"  {label:38s} | {fmt(u):>7s} | {fmt_c(c):>7s} | {fmt(nl,4):>7s} | {fmt(rl,4):>7s} | {fmt(dl,4):>9s} | {fmt(m93,4):>7s} | {hl:6d}")
        table[k] = {"utmos": u, "cer": c, "norm_l1": nl, "raw_l1": rl, "dtw_raw_l1": dl, "mod93": m93, "holes": hl}
    return table


def section_f_chunk_attn_benchmark(model, device: torch.device):
    print("\n" + "=" * 78)
    print("[F] ChunkLookaheadBackbone 零循环并行 + 超长流式局部 RoPE 等价验证 & 速度实测")
    print("=" * 78)
    if model.backbone_type != "chunk_attn":
        return {}
    bb = model.backbone
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    bb_f32 = bb.float().eval()  # a9: eval 态屏蔽 dropout 以精确校验前反向图数值等价
    torch.manual_seed(123)
    T_lens = [75, 43, 19, 3]
    B_t, T_t, D = len(T_lens), max(T_lens), model.d_model
    mask_t = torch.arange(T_t, device=device)[None, :] < torch.tensor(T_lens, device=device)[:, None]
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

    bb_f32.eval()
    with torch.no_grad():
        xs = torch.randn(1, 605, D, device=device)
        y_off = bb_f32.forward(xs)
        st = bb_f32.init_state(1, device, xs.dtype)
        outs = []
        for s in range(0, 605, 4):
            o, st = bb_f32.forward_step(xs[:, s : s + 4], st, is_final=(s + 4 >= 605))
            if o.shape[1] > 0:
                outs.append(o)
        stream_diff = (y_off - torch.cat(outs, dim=1)).abs().max().item()
        xs_pert = xs.clone()
        xs_pert[:, 12 + bb_f32.lookahead :] = torch.randn(1, 605 - 12 - bb_f32.lookahead, D, device=device)
        leak = (y_off[:, :12] - bb_f32.forward(xs_pert)[:, :12]).abs().max().item()

    print(f"  [数值等价 T=605] Fwd={fwd_diff:.2e} | xGrad={xgrad_diff:.2e} | pGrad={pgrad_diff:.2e} | Stream={stream_diff:.2e} | Leak={leak:.1e}")
    assert fwd_diff < 1e-4 and xgrad_diff < 1e-3 and pgrad_diff < 1e-3 and stream_diff < 1e-4 and leak == 0.0
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return {"fwd_diff": fwd_diff, "xgrad_diff": xgrad_diff, "pgrad_diff": pgrad_diff, "stream_diff_T605": stream_diff, "leak": leak}


def main():
    ap = argparse.ArgumentParser(description="AetherTTS Phase 1 (a9) 深度诊断脚本")
    ap.add_argument("--root", default="./aethertts_exp")
    ap.add_argument("--profile", default="standard_phase1")
    ap.add_argument("--dataset", default="bznsyp")
    ap.add_argument("--model", default="E3_chunk_attn_C4_L8")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--vocoder-ckpt", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--num-utts", type=int, default=5)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = root / "diag_out" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_checkpoint(root, args.model, args.profile, args.dataset, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    stats = ckpt["stats"]
    bb_type = ckpt.get("backbone_type", "chunk_attn")
    lookahead = int(ckpt.get("lookahead", 8))

    model = lib.AetherAcousticModel(
        vocab_size=len(ckpt.get("vocab", lib.FRONTEND.vocab)),
        backbone_type=bb_type,
        d_model=int(cfg.get("d_model", 256)),
        n_mels=int(cfg.get("n_mels", 100)),
        k_left=int(cfg.get("k_left", 3)),
        lookahead=lookahead,
        swa_window=int(cfg.get("swa_window", 48)),
        chunk_size=int(cfg.get("chunk_attn_size", 4)),
        frame_lookahead=int(cfg.get("frame_lookahead", 2)),
        dropout=float(cfg.get("dropout_acoustic", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[Load] 声学权重加载成功: {ckpt_path} (arch={ckpt.get('arch_tag')}, step={ckpt.get('step')}, best_val_mel={ckpt.get('best_val_mel')}, best_val_infer={ckpt.get('best_val_infer')})")

    manifest_path = find_manifest(root, args.dataset, cfg.get("max_train_hours"), args.cache_dir)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent

    causal_voc = None
    voc_ckpt = find_vocoder_ckpt(root, args.profile, args.dataset, args.vocoder_ckpt)
    if voc_ckpt is not None:
        is_legacy_a9 = (ckpt.get("arch_tag") == "a9") or ("_a9" in voc_ckpt.parent.name.lower() and "_a10" not in voc_ckpt.parent.name.lower())
        n_harm = 24 if is_legacy_a9 else int(cfg.get("vocoder_n_harmonics", 48))
        use_chirp = False if is_legacy_a9 else bool(cfg.get("vocoder_chirp", True))
        causal_voc = lib.CausalVocosLite(
            n_mels=int(cfg.get("n_mels", 100)),
            dim=int(cfg.get("vocoder_dim", 320)),
            n_fft=int(cfg.get("n_fft", 1024)),
            hop_length=int(cfg.get("hop_length", 256)),
            lookahead=int(cfg.get("vocoder_lookahead", 4)),
            sample_rate=int(cfg.get("sample_rate", 24000)),
            n_harmonics=n_harm,
            use_chirp=use_chirp,
        ).to(device)
        causal_voc.load_state_dict(torch.load(voc_ckpt, map_location=device, weights_only=True), strict=True)
        causal_voc.eval()
        print(f"[Load] 谐波源因果声码器: {voc_ckpt} (H={n_harm}, chirp={use_chirp})")

    pretrained_vocos = None
    try:
        from vocos import Vocos
        pretrained_vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    except Exception:
        pass

    report = {
        "ckpt": str(ckpt_path),
        "step": ckpt.get("step"),
        "best_val_mel": ckpt.get("best_val_mel"),
        "history": section_a_history(root),
        "loss_split": section_b_train_val_loss(model, meta, manifest_dir, stats, int(cfg.get("n_mels", 100)), device),
        "aligner": section_c_aligner_forensics(model, lib.FRONTEND, meta, manifest_dir, stats, int(cfg.get("n_mels", 100)), device, out_dir, num_utts=args.num_utts),
        "frontend": section_d_frontend_audit(lib.FRONTEND, meta, manifest_dir, max_utts=30),
        "isolation": section_e_link_isolation(
            model, lib.FRONTEND, causal_voc, pretrained_vocos, meta, manifest_dir, stats,
            int(cfg.get("n_mels", 100)), int(cfg.get("sample_rate", 24000)), device, out_dir,
            num_utts=args.num_utts, run_utmos=(not args.skip_utmos), run_asr=(not args.skip_asr),
            whisper_name=args.whisper_model,
        ),
    }
    if not args.skip_bench:
        report["chunk_attn_bench"] = section_f_chunk_attn_benchmark(model, device)
    summary_path = out_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[Done] 完整诊断报告已保存至: {summary_path}")


if __name__ == "__main__":
    main()
