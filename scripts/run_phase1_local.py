#!/usr/bin/env python3
"""
自建 Linux GPU 服务器（如 RTX 4090 / AutoDL）无界面命令行运行入口。
直接顺序执行 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb，
全部数据、断点、权重、音频与评测报告均保存在本地 --root 目录（不依赖 Google Drive）。

示例（推荐配合 tmux 后台离线运行）：
    tmux new -s aether
    python scripts/run_phase1_local.py --profile standard_phase1 --dataset bznsyp --root ./aethertts_exp
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run AetherTTS Phase 1 experiment notebook locally on Linux GPU (e.g. RTX 4090).")
    p.add_argument("--notebook", default="notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb", help="Path to the experiment notebook.")
    p.add_argument("--root", default="./aethertts_exp", help="Local directory to store datasets, checkpoints, samples, and reports.")
    p.add_argument("--profile", default="standard_phase1", choices=["quick_smoke", "standard_phase1", "full_quality"], help="Experiment duration/data budget profile.")
    p.add_argument("--dataset", default="bznsyp", choices=["bznsyp", "ljspeech", "custom_bilingual"], help="Training dataset source.")
    p.add_argument("--vocoder", default="both", choices=["both", "pretrained_vocos", "causal_vocos_lite"], help="Vocoder mode.")
    p.add_argument("--models", default="", help="Optional comma-separated subset of models, e.g. E1_conv_unigru_L4,E2_swa_L4")
    p.add_argument("--backup-steps", type=int, default=1000, help="Save resumable latest.pt every N acoustic steps (default: 1000).")
    p.add_argument("--backup-minutes", type=float, default=8.0, help="Also force-save latest.pt every M minutes (default: 8.0).")
    p.add_argument("--skip-pip", action="store_true", help="Skip automatic pip install of missing dependencies.")
    return p.parse_args()


def main():
    args = parse_args()
    nb_path = Path(args.notebook).expanduser().resolve()
    if not nb_path.exists():
        print(f"[Error] Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(1)

    os.environ["AETHER_ENV_MODE"] = "local"
    os.environ["AETHER_ROOT"] = str(Path(args.root).expanduser().resolve())
    os.environ["AETHER_PROFILE"] = args.profile
    os.environ["AETHER_DATASET"] = args.dataset
    os.environ["AETHER_VOCODER"] = args.vocoder
    os.environ["AETHER_BACKUP_STEPS"] = str(args.backup_steps)
    os.environ["AETHER_BACKUP_MINUTES"] = str(args.backup_minutes)
    if args.models:
        os.environ["AETHER_MODELS"] = args.models
    if args.skip_pip:
        os.environ["AETHER_SKIP_PIP"] = "1"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    # 直接在 __main__.__dict__ 命名空间执行，保证 PyTorch DataLoader 多进程 worker 无论 fork/spawn 都能正常 pickle 顶层类与函数
    ns = sys.modules["__main__"].__dict__
    ns["__file__"] = str(nb_path)

    code_idx = 0
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_idx += 1
        src_lines = cell.get("source", [])
        src = "".join(src_lines) if isinstance(src_lines, list) else str(src_lines)
        # 过滤 Jupyter 魔术行（若有）
        filtered = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        print(f"\n{'#' * 72}\n# Executing Notebook Code Cell #{code_idx} (cell index {cell_idx})\n{'#' * 72}", flush=True)
        exec(compile(filtered, f"{nb_path.name}#cell_{cell_idx}", "exec"), ns)

    print("\n[All Done] 本地实验全部执行完毕！", flush=True)


if __name__ == "__main__":
    main()
