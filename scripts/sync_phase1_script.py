#!/usr/bin/env python3
"""
从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb 确定性生成：
  1. scripts/aethertts_lib.py    — 无副作用纯定义库（前端 / DSP / ChunkAttn 主干 / Aligner / 声学模型 / 谐波源因果声码器），供诊断与推理直接 import
  2. scripts/aethertts_phase1.py — 独立 Python CLI 训练与评测运行器（无需 Notebook / 无 exec-cell）

保证 Notebook、独立训练脚本与诊断工具永远保持单一真相源、零架构漂移。

用法：
    python scripts/sync_phase1_script.py          # 重新生成 scripts/aethertts_lib.py 与 scripts/aethertts_phase1.py
    python scripts/sync_phase1_script.py --check  # 仅校验是否已同步，不同步时以非零退出
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "AetherTTS_Colab_Phase1_Experiment.ipynb"
OUT_PATH = REPO_ROOT / "scripts" / "aethertts_phase1.py"
LIB_PATH = REPO_ROOT / "scripts" / "aethertts_lib.py"

LIB_HEADER = '''#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a9) — 纯模型与前端基础设施定义库（import 零副作用）。

由 scripts/sync_phase1_script.py 从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb
的 Cell 3 与 Cell 5 确定性生成（请勿手改本文件，修改 Notebook 后运行 sync_phase1_script.py 即可同步）。
"""
import json
import math
import os
import random
import shutil
import signal
import sys
import tarfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ENV_MODE = os.environ.get("AETHER_ENV_MODE", "local")
IS_COLAB = (ENV_MODE == "colab")
IS_LOCAL = (ENV_MODE == "local")
IS_KAGGLE = (ENV_MODE == "kaggle")
STORAGE_ROOT = Path(os.environ.get("AETHER_ROOT", "./aethertts_exp")).expanduser().resolve()
LOCAL_WORK = STORAGE_ROOT
DRIVE_ROOT = STORAGE_ROOT
_VOC_LA_DEFAULT = int(os.environ.get("AETHER_VOCODER_LOOKAHEAD", "4"))
CFG = {
    "seed": 42,
    "sample_rate": 24000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 100,
    "min_audio_sec": 0.8,
    "max_audio_sec": 12.0,
    "d_model": 256,
    "k_left": 3,
    "swa_window": 48,
    "chunk_attn_size": 4,
    "stream_text_chunk": 4,
    "stream_mel_first_chunk": max(6, _VOC_LA_DEFAULT + 4),
    "stream_mel_chunk": 16,
    "dropout_acoustic": float(os.environ.get("AETHER_DROPOUT", "0.1")),
    "vocoder_dim": int(os.environ.get("AETHER_VOCODER_DIM", "320")),
    "frame_lookahead": int(os.environ.get("AETHER_FRAME_LOOKAHEAD", "2")),
    "vocoder_lookahead": _VOC_LA_DEFAULT,
    "profile": os.environ.get("AETHER_PROFILE", "standard_phase1"),
    "dataset_source": os.environ.get("AETHER_DATASET", "bznsyp"),
}
'''

HEADER = '''#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a9) — 独立 Python 运行器（无需 Notebook / 无 exec-cell）。

由 scripts/sync_phase1_script.py 从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb
确定性 1:1 生成（请勿手改本文件，修改 Notebook 后运行 sync_phase1_script.py 即可同步）。

默认聚焦唯一主力路线 E3_chunk_attn_C4_L8（5 层深双向分块前瞻 ChunkLookaheadBackbone + 局部窗口相对 RoPE +
防坍缩单调对齐 + 共享帧前瞻与多尺度空洞 F0/UV/帧解码器 + 连续相位谐波源-滤波器 CausalVocosLite + 相位一致性损失 + GT-F0 对齐 GTA 闭环精调），
同时保留 E0/E1/E2/E4 代码供归档对照（可通过 --models 显式指定）。

用法：
    python scripts/aethertts_phase1.py --profile full_quality --dataset bznsyp --root ./aethertts_exp
    python scripts/aethertts_phase1.py --selftest          # 验证前端零 UNK + 防坍缩对齐 + 长流式局部 RoPE + 谐波 STFT COLA + 相位损失/声码器流式 + autocast 精度门禁
    python scripts/aethertts_phase1.py --profile-steps 20  # 打印单步训练耗时分解后退出
"""
import argparse

_AP = argparse.ArgumentParser(description="AetherTTS Phase 1 (a9) standalone runner (synced from notebook).")
_AP.add_argument("--root", default="./aethertts_exp", help="Local directory for datasets, checkpoints, samples, reports.")
_AP.add_argument("--profile", default="standard_phase1", choices=["quick_smoke", "standard_phase1", "full_quality"])
_AP.add_argument("--dataset", default="bznsyp", choices=["bznsyp", "ljspeech", "custom_bilingual"])
_AP.add_argument("--vocoder", default="both", choices=["both", "pretrained_vocos", "causal_vocos_lite"])
_AP.add_argument("--models", default="", help="Optional comma-separated model override (default: only E3_chunk_attn_C4_L8).")
_AP.add_argument("--frame-lookahead", type=int, default=None, help="声学 frame_pre 共享帧级右前瞻帧数（默认 2 帧；0 表示关闭）.")
_AP.add_argument("--vocoder-lookahead", type=int, default=None, help="CausalVocosLite 集中式右前瞻帧数（a9 默认 4 帧，覆盖完整 1024 点 iSTFT 右半窗）.")
_AP.add_argument("--vocoder-dim", type=int, default=None, help="CausalVocosLite 通道宽度（a9 默认 320）.")
_AP.add_argument("--dropout", type=float, default=None, help="声学模型 Dropout 比例（a9 默认 0.1）.")
_AP.add_argument("--batch-size", type=int, default=None, help="可选覆写声学训练 batch_size（如 RTX 4090 覆写为 64）.")
_AP.add_argument("--gta-prob", type=float, default=None, help="声码器 Stage 2 末 25% 混入 GT-F0 对齐 GTA 预测 Mel 的概率（a9 默认 0.25；0 表示关闭）.")
_AP.add_argument("--backup-steps", type=int, default=1000)
_AP.add_argument("--backup-minutes", type=float, default=8.0)
_AP.add_argument("--skip-pip", action="store_true")
_AP.add_argument("--selftest", action="store_true", help="Run frontend/aligner/long-streaming/vocoder gates, then exit.")
_AP.add_argument("--profile-steps", type=int, default=0, help="If >0, time N training steps component-wise, print breakdown, then exit.")
_ARGS = _AP.parse_args()

import os as _os
from pathlib import Path as _Path
_os.environ["AETHER_ENV_MODE"] = "local"
_os.environ["AETHER_ROOT"] = str(_Path(_ARGS.root).expanduser().resolve())
_os.environ["AETHER_PROFILE"] = _ARGS.profile
_os.environ["AETHER_DATASET"] = _ARGS.dataset
_os.environ["AETHER_VOCODER"] = _ARGS.vocoder
_os.environ["AETHER_BACKUP_STEPS"] = str(_ARGS.backup_steps)
_os.environ["AETHER_BACKUP_MINUTES"] = str(_ARGS.backup_minutes)
if _ARGS.models:
    _os.environ["AETHER_MODELS"] = _ARGS.models
if _ARGS.frame_lookahead is not None:
    _os.environ["AETHER_FRAME_LOOKAHEAD"] = str(_ARGS.frame_lookahead)
if _ARGS.vocoder_lookahead is not None:
    _os.environ["AETHER_VOCODER_LOOKAHEAD"] = str(_ARGS.vocoder_lookahead)
if _ARGS.vocoder_dim is not None:
    _os.environ["AETHER_VOCODER_DIM"] = str(_ARGS.vocoder_dim)
if _ARGS.dropout is not None:
    _os.environ["AETHER_DROPOUT"] = str(_ARGS.dropout)
if _ARGS.batch_size is not None:
    _os.environ["AETHER_BATCH_SIZE"] = str(_ARGS.batch_size)
if _ARGS.gta_prob is not None:
    _os.environ["AETHER_GTA_PROB"] = str(_ARGS.gta_prob)
if _ARGS.skip_pip:
    _os.environ["AETHER_SKIP_PIP"] = "1"
SELFTEST = _ARGS.selftest
PROFILE_STEPS = _ARGS.profile_steps
'''

ALIGNER_SELFTEST_BLOCK = '''
# ==============================================================================
# 6b. 向量化热路径 vs 参考实现 逐位等价自检（--selftest）
# ==============================================================================
def run_aligner_equivalence_selftest():
    print("=" * 72)
    print("向量化对齐热路径 vs 参考实现 —— 逐位等价自检 (torch.equal)")
    print("=" * 72)
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        B = 6
        F_max = int(torch.randint(24, 90, (1,), generator=g))
        T_max = int(torch.randint(6, 30, (1,), generator=g))
        mel_lens = torch.randint(max(2, F_max // 2), F_max + 1, (B,), generator=g)
        text_lens = torch.randint(2, T_max + 1, (B,), generator=g)
        text_lens[0] = 1  # 单 token 边界用例
        lp = torch.randn(B, F_max, T_max, generator=g) * 3.0
        tmask = torch.arange(T_max)[None, :] < text_lens[:, None]
        lp = lp.masked_fill(~tmask[:, None, :], -1e4)
        h_ref, d_ref = viterbi_monotonic_alignment_ref(lp, text_lens, mel_lens)
        h_fast, d_fast = viterbi_monotonic_alignment(lp, text_lens, mel_lens)
        assert torch.equal(h_ref, h_fast), f"viterbi attn_hard mismatch seed={seed}"
        assert torch.equal(d_ref, d_fast), f"viterbi durations mismatch seed={seed}"
        p_ref = build_beta_binomial_log_prior_ref(text_lens, mel_lens, T_max, F_max)
        p_fast = build_beta_binomial_log_prior(text_lens, mel_lens, T_max, F_max)
        assert torch.equal(p_ref, p_fast), f"beta-binomial prior mismatch seed={seed}"
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            lp_d, tl_d, ml_d = lp.to(dev), text_lens.to(dev), mel_lens.to(dev)
            assert torch.equal(viterbi_monotonic_alignment_ref(lp_d, tl_d, ml_d)[0].cpu(), h_ref)
            assert torch.equal(viterbi_monotonic_alignment_ref(lp_d, tl_d, ml_d)[1].cpu(), d_ref)
            p_ref_d = build_beta_binomial_log_prior_ref(tl_d, ml_d, T_max, F_max)
            p_fast_d = build_beta_binomial_log_prior(tl_d, ml_d, T_max, F_max)
            assert torch.equal(p_ref_d.cpu(), p_fast_d.cpu()), f"prior cuda mismatch seed={seed}"
            assert torch.allclose(p_ref_d.cpu(), p_ref, rtol=1e-5, atol=1e-4), f"prior cross-device drift seed={seed}"
        print(f"  [PASS] seed={seed}  shape=({B},{F_max},{T_max}) CPU+CUDA 逐位一致")
    print("[SelfTest] 全部通过：向量化实现与参考实现完全等价。")
    print("=" * 72)


if SELFTEST:
    run_aligner_equivalence_selftest()
'''

SELFTEST_EXIT_BLOCK = '''
if SELFTEST:
    print("[SelfTest] 前端音系、长流式局部 RoPE、零前瞻膨胀、谐波源因果声码器、autocast 精度及保留主干门禁全部通过，退出 --selftest 模式。")
    import sys as _sys
    _sys.exit(0)
'''

STEP_PROFILE_BLOCK = '''
# ==============================================================================
# 7b. 单步耗时分解（--profile-steps N）：定位数据/前向/对齐/反向各占多少
# ==============================================================================
def _acc_store(acc, k, dt):
    acc[k] = acc.get(k, 0.0) + dt


def _run_step_profile():
    if PROFILE_STEPS <= 0:
        return
    exp_name, bb_type, la = EXPERIMENTS[0]
    model = AetherAcousticModel(
        vocab_size=FRONTEND.vocab_size, backbone_type=bb_type, d_model=CFG["d_model"],
        n_mels=CFG["n_mels"], k_left=CFG["k_left"], lookahead=la,
        swa_window=CFG["swa_window"], chunk_size=CFG["chunk_attn_size"],
        frame_lookahead=CFG["frame_lookahead"],
    ).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr_acoustic"])
    pids_all, lids_all, bids_all, tl_all, mels_all, f0_all, e_all, uv_all, ml_all = next(iter(train_loader))
    acc, n_warm = {}, 2
    for it in range(PROFILE_STEPS + n_warm):
        if it == n_warm:
            acc = {}
        t0 = time.perf_counter()
        pids, lids, bids, t_lens = pids_all.to(DEVICE), lids_all.to(DEVICE), bids_all.to(DEVICE), tl_all.to(DEVICE)
        mels, f0s, es, uvs, m_lens = mels_all.to(DEVICE), f0_all.to(DEVICE), e_all.to(DEVICE), uv_all.to(DEVICE), ml_all.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        _acc_store(acc, "h2d", time.perf_counter() - t0)
        opt.zero_grad(set_to_none=True)
        tf0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            out = model.forward_train(pids, lids, t_lens, mels, f0s, es, m_lens, step_ratio=0.5, boundary_ids=bids, uv_gt=uvs)
            loss = out["loss"]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        _acc_store(acc, "fwd(含对齐)", time.perf_counter() - tf0)
        tb0 = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        _acc_store(acc, "bwd+opt", time.perf_counter() - tb0)
        _acc_store(acc, "e2e", time.perf_counter() - t0)
    n = max(1, PROFILE_STEPS)
    print(f"[Profile] model={exp_name} batch={pids.shape[0]} T={pids.shape[1]} F={mels.shape[-1]} | 每步平均耗时 ({n / max(acc['e2e'], 1e-6):.2f} it/s):")
    for k in ("h2d", "fwd(含对齐)", "bwd+opt", "e2e"):
        print(f"  {k:12s}: {acc[k] / n * 1000:8.1f} ms")
    if bb_type == "chunk_attn":
        text_mask = torch.arange(pids.shape[1], device=pids.device)[None, :] < t_lens[:, None]
        x_in = model.embed_text(pids, lids, text_mask, boundary_ids=bids).detach()
        def _bench_bb(fn, runs=5):
            for _ in range(2):
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                    fn(x_in, mask=text_mask).sum().backward()
            model.backbone.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_b = time.perf_counter()
            for _ in range(runs):
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                    fn(x_in, mask=text_mask).sum().backward()
                model.backbone.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t_b) / runs * 1000.0
        ms_par = _bench_bb(model.backbone.forward)
        ms_seq = _bench_bb(model.backbone._forward_sequential)
        print(f"  {'bb(parallel)':14s}: {ms_par:8.1f} ms  (fwd+bwd)")
        print(f"  {'bb(seq_ref)':14s}: {ms_seq:8.1f} ms  (fwd+bwd) -> backbone 提速 {ms_seq / max(ms_par, 1e-6):.2f}x")
    with torch.no_grad():
        text_mask = torch.arange(pids.shape[1], device=pids.device)[None, :] < t_lens[:, None]
        mel_mask = torch.arange(mels.shape[-1], device=pids.device)[None, :] < m_lens[:, None]
        x_emb = model.embed_text(pids, lids, text_mask, boundary_ids=bids)
        prior = build_beta_binomial_log_prior(t_lens, m_lens, pids.shape[1], mels.shape[-1])
        attn_logprob, _ = model.aligner(mels, x_emb, text_mask, attn_log_prior=prior, mel_mask=mel_mask)
        for tag, fn in (("viterbi(fast)", viterbi_monotonic_alignment), ("viterbi(ref)", viterbi_monotonic_alignment_ref),
                        ("prior(fast)", lambda a, b, c, d: build_beta_binomial_log_prior(a, b, c, d)),
                        ("prior(ref)", lambda a, b, c, d: build_beta_binomial_log_prior_ref(a, b, c, d))):
            t_p = time.perf_counter()
            for _ in range(5):
                fn(attn_logprob, t_lens, m_lens) if "viterbi" in tag else fn(t_lens, m_lens, pids.shape[1], mels.shape[-1])
            print(f"  {tag:14s}: {(time.perf_counter() - t_p) / 5 * 1000:8.1f} ms")
    print("[Profile] 完成（--profile-steps 模式仅测量，不写任何 checkpoint）。")
    sys.exit(0)


_run_step_profile()
'''

# 单元重排：
# 1) 把 Cell 5 (模型) 和 Cell 6 (门禁) 提前到 Cell 4 (数据下载) 之前，使 --selftest 无需下载数据；
# 2) 先跑 Cell 8 (声学模型 E3 训练)，再跑 Cell 9 (导出 GTA Mel + 因果谐波源声码器闭环训练)！
CELL_PLAN = [
    ("cc414d72", "Cell 1: 环境识别与全局配置"),
    ("328dae20", "Cell 2: 依赖安装"),
    ("609ce47a", "Cell 3: 基础设施（Mel/F0+UV/中英音系与韵律前端）"),
    ("da9fbbaa", "Cell 5: 核心模型（ChunkLookaheadBackbone 主力 + 保留主干 + 浊音感知声学模型 + 谐波源因果声码器）"),
    ("7f900006", "Cell 6: 训练前硬门禁（前端音系 + 长流式局部 RoPE + 零前瞻膨胀 + 声码器流式等价 + autocast 精度）"),
    ("439d2250", "Cell 4: 数据下载与 v4 预处理"),
    ("09610ad7", "Cell 7: DataLoader"),
    ("1f7a318d", "Cell 8: 声学主干训练（默认唯一主力 E3_chunk_attn_C4_L8）"),
    ("2df0111e", "Cell 9: GTA 预测谱导出与谐波源因果声码器闭环训练"),
    ("86b0212b", "Cell 10: 评测音频合成（含动态低延迟首包流式调度）"),
    ("9b7913a1", "Cell 11: 自动打分、全链路 INT8 校验与 CPU 基准"),
    ("65e43d36", "Cell 12: 可视化与试听打分表"),
    ("41ac40af", "Cell 13: 可选教师造数（默认跳过）"),
]

LIB_CELLS = [
    ("609ce47a", "Cell 3: 基础设施（Mel/F0+UV/中英音系与韵律前端）"),
    ("da9fbbaa", "Cell 5: 核心模型（ChunkLookaheadBackbone 主力 + 保留主干 + 浊音感知声学模型 + 谐波源因果声码器）"),
]


def _load_nb_cells() -> dict:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return {c.get("id"): "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"}


def render_lib() -> str:
    by_id = _load_nb_cells()
    parts = [LIB_HEADER.rstrip("\n")]
    for cid, title in LIB_CELLS:
        if cid not in by_id:
            raise KeyError(f"Notebook missing expected code cell id={cid!r} ({title})")
        code = by_id[cid].rstrip("\n")
        # 库模块被 import 时静默初始化，不打印前端演示日志
        code = code.replace(
            'if True:  # FRONTEND_DEMO_PRINT',
            'if __name__ == "__main__":  # FRONTEND_DEMO_PRINT',
        )
        banner = (
            "\n\n# " + "#" * 77 + "\n"
            f"# {title}\n"
            "# " + "#" * 77 + "\n"
        )
        parts.append(banner + code)
    parts.append("\n")
    return "".join(parts)


def render_script() -> str:
    by_id = _load_nb_cells()
    parts = [HEADER.rstrip("\n")]
    for cid, title in CELL_PLAN:
        if cid not in by_id:
            raise KeyError(f"Notebook missing expected code cell id={cid!r} ({title})")
        code = by_id[cid].rstrip("\n")
        if cid == "41ac40af":
            code = code.replace(
                "RUN_TEACHER_SYNTH = False  # Phase 2 造数时改为 True",
                'RUN_TEACHER_SYNTH = os.environ.get("AETHER_TEACHER_SYNTH", "0") == "1"',
            )
        banner = (
            "\n\n# " + "#" * 77 + "\n"
            f"# {title}\n"
            "# " + "#" * 77 + "\n"
        )
        parts.append(banner + code)
        if cid == "da9fbbaa":
            parts.append("\n" + ALIGNER_SELFTEST_BLOCK.strip("\n"))
        elif cid == "7f900006":
            parts.append("\n" + SELFTEST_EXIT_BLOCK.strip("\n"))
        elif cid == "09610ad7":
            parts.append("\n" + STEP_PROFILE_BLOCK.strip("\n"))
    parts.append('\n\nprint("[All Done] 独立 Python 实验全部执行完毕！", flush=True)\n')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Exit non-zero if generated scripts are out of sync.")
    args = ap.parse_args()
    rendered_script = render_script()
    rendered_lib = render_lib()
    if args.check:
        cur_script = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        cur_lib = LIB_PATH.read_text(encoding="utf-8") if LIB_PATH.exists() else ""
        if cur_script != rendered_script or cur_lib != rendered_lib:
            print(
                f"[OUT OF SYNC] {OUT_PATH.name} or {LIB_PATH.name} differs from {NB_PATH.name}. "
                f"Run: python scripts/sync_phase1_script.py",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[OK] {OUT_PATH.name} and {LIB_PATH.name} are 100% in sync with {NB_PATH.name}.")
        return
    LIB_PATH.write_text(rendered_lib, encoding="utf-8")
    OUT_PATH.write_text(rendered_script, encoding="utf-8")
    print(
        f"[Synced] Generated {LIB_PATH.name} ({len(rendered_lib.splitlines())} lines) and "
        f"{OUT_PATH.name} ({len(rendered_script.splitlines())} lines) from {NB_PATH.name}."
    )


if __name__ == "__main__":
    main()
