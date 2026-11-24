#!/usr/bin/env python3
"""
从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb 确定性生成独立 Python 运行器
scripts/aethertts_phase1.py，保证 Notebook 与独立脚本永远保持单一真相源、零架构漂移。

用法：
    python scripts/sync_phase1_script.py          # 重新生成 scripts/aethertts_phase1.py
    python scripts/sync_phase1_script.py --check  # 仅校验是否已同步，不同步时以非零退出
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "AetherTTS_Colab_Phase1_Experiment.ipynb"
OUT_PATH = REPO_ROOT / "scripts" / "aethertts_phase1.py"

HEADER = '''#!/usr/bin/env python3
"""
AetherTTS Phase 1 — 独立 Python 运行器（无需 Notebook / 无 exec-cell）。

由 scripts/sync_phase1_script.py 从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb
确定性 1:1 生成（请勿手改本文件，修改 Notebook 后运行 sync_phase1_script.py 即可同步）。

与 Notebook / scripts/run_phase1_local.py 共享同一套 a4 流式输入架构代码（含 E0 离线、E1 L0/L8、
E2 SWA、E3 5 层深双向分块前瞻 ChunkLookaheadBackbone、E4 标准门控 DeltaNet、音素内多尺度相位、
帧解码器 2 帧前瞻、声码器 2 帧前瞻及批向量化 Beta-Binomial/Viterbi 对齐），可随时无缝互相续训。

用法：
    python scripts/aethertts_phase1.py --profile standard_phase1 --dataset bznsyp --root ./aethertts_exp
    python scripts/aethertts_phase1.py --selftest          # 验证向量化对齐逐位等价 + 流式/零前瞻膨胀/梯度门禁
    python scripts/aethertts_phase1.py --profile-steps 20  # 打印单步训练耗时分解后退出
"""
import argparse

_AP = argparse.ArgumentParser(description="AetherTTS Phase 1 standalone runner (synced from notebook).")
_AP.add_argument("--root", default="./aethertts_exp", help="Local directory for datasets, checkpoints, samples, reports.")
_AP.add_argument("--profile", default="standard_phase1", choices=["quick_smoke", "standard_phase1", "full_quality"])
_AP.add_argument("--dataset", default="bznsyp", choices=["bznsyp", "ljspeech", "custom_bilingual"])
_AP.add_argument("--vocoder", default="both", choices=["both", "pretrained_vocos", "causal_vocos_lite"])
_AP.add_argument("--models", default="", help="Optional comma-separated subset, e.g. E1_conv_unigru_L8,E3_chunk_attn_C4_L8,E4_gated_deltanet_L8")
_AP.add_argument("--frame-lookahead", type=int, default=None, help="声学 frame_decoder 首层集中式右前瞻帧数（默认 2 帧；0 表示关闭）.")
_AP.add_argument("--vocoder-lookahead", type=int, default=None, help="CausalVocosLite 集中式右前瞻帧数（默认 2 帧；0 表示关闭前瞻）.")
_AP.add_argument("--backup-steps", type=int, default=1000)
_AP.add_argument("--backup-minutes", type=float, default=8.0)
_AP.add_argument("--skip-pip", action="store_true")
_AP.add_argument("--selftest", action="store_true", help="Run aligner bit-equality + streaming/zero-explosion/gradient gates, then exit.")
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
    print("[SelfTest] 架构、零前瞻膨胀、多块流式等价与强遗忘梯度门禁全部通过，退出 --selftest 模式。")
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
    pids_all, lids_all, tl_all, mels_all, f0_all, e_all, ml_all = next(iter(train_loader))
    acc, n_warm = {}, 2
    for it in range(PROFILE_STEPS + n_warm):
        if it == n_warm:
            acc = {}
        t0 = time.perf_counter()
        pids, lids, t_lens = pids_all.to(DEVICE), lids_all.to(DEVICE), tl_all.to(DEVICE)
        mels, f0s, es, m_lens = mels_all.to(DEVICE), f0_all.to(DEVICE), e_all.to(DEVICE), ml_all.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        _acc_store(acc, "h2d", time.perf_counter() - t0)
        opt.zero_grad(set_to_none=True)
        tf0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            out = model.forward_train(pids, lids, t_lens, mels, f0s, es, m_lens, step_ratio=0.5)
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
    print(f"[Profile] model={exp_name} batch={pids.shape[0]} T={pids.shape[1]} F={mels.shape[-1]} | 每步平均耗时:")
    for k in ("h2d", "fwd(含对齐)", "bwd+opt", "e2e"):
        print(f"  {k:12s}: {acc[k] / n * 1000:8.1f} ms")
    with torch.no_grad():
        text_mask = torch.arange(pids.shape[1], device=pids.device)[None, :] < t_lens[:, None]
        x_emb = model.embed_text(pids, lids, text_mask)
        prior = build_beta_binomial_log_prior(t_lens, m_lens, pids.shape[1], mels.shape[-1])
        attn_logprob, _ = model.aligner(mels, x_emb, text_mask, attn_log_prior=prior)
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

# 单元重排：把 Cell 5 (模型) 和 Cell 6 (门禁) 提前到 Cell 4 (数据下载) 之前，使 --selftest 无需下载数据
CELL_PLAN = [
    ("cc414d72", "Cell 1: 环境识别与全局配置"),
    ("328dae20", "Cell 2: 依赖安装"),
    ("609ce47a", "Cell 3: 基础设施（Mel/F0/前端）"),
    ("da9fbbaa", "Cell 5: 核心模型（E0-E4 流式输入主干 + 向量化对齐 + 有限前瞻帧解码器/声码器）"),
    ("7f900006", "Cell 6: 训练前硬门禁（零前瞻膨胀 + 多块流式等价 + 强遗忘梯度校验）"),
    ("439d2250", "Cell 4: 数据下载与预处理"),
    ("09610ad7", "Cell 7: DataLoader"),
    ("2df0111e", "Cell 8: 声码器训练"),
    ("1f7a318d", "Cell 9: 声学主干训练"),
    ("86b0212b", "Cell 10: 评测音频合成"),
    ("9b7913a1", "Cell 11: 自动打分与 CPU 基准"),
    ("65e43d36", "Cell 12: 可视化与盲听表"),
    ("41ac40af", "Cell 13: 可选教师造数（默认跳过）"),
]


def render_script() -> str:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    by_id = {c.get("id"): "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"}
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
    ap.add_argument("--check", action="store_true", help="Exit non-zero if scripts/aethertts_phase1.py is out of sync.")
    args = ap.parse_args()
    rendered = render_script()
    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != rendered:
            print(f"[OUT OF SYNC] {OUT_PATH} differs from {NB_PATH}. Run: python scripts/sync_phase1_script.py", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] {OUT_PATH} is 100% in sync with {NB_PATH}.")
        return
    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"[Synced] Generated {OUT_PATH} ({len(rendered.splitlines())} lines) from {NB_PATH}.")


if __name__ == "__main__":
    main()
