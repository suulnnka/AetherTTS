#!/usr/bin/env python3
"""
AetherTTS Phase 1 — 独立 Python 运行器（无需 Notebook / 无 exec-cell）。

从 notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb 1:1 移植，并针对自建 4090 上的
CPU 瓶颈做了两处「逐位等价」的热路径向量化（消除逐样本 Python 循环）：

  1. viterbi_monotonic_alignment      批向量化单调 Viterbi（参考实现保留为 *_ref）
  2. build_beta_binomial_log_prior    批向量化 Beta-Binomial 对角先验（参考实现保留为 *_ref）

两者与 notebook 参考实现逐位一致（--selftest 可随时验证），因此 checkpoint、目录结构与
scripts/run_phase1_local.py 完全兼容：notebook 跑到一半的断点可以直接用本脚本精确续训，
反之亦然。

用法：
    python scripts/aethertts_phase1.py --profile standard_phase1 --dataset bznsyp --root ./aethertts_exp
    python scripts/aethertts_phase1.py --selftest          # 验证向量化与参考实现逐位等价 + 架构门禁
    python scripts/aethertts_phase1.py --profile-steps 20  # 打印单步训练耗时分解后退出
"""
import argparse

_AP = argparse.ArgumentParser(description="AetherTTS Phase 1 standalone runner (no notebook).")
_AP.add_argument("--root", default="./aethertts_exp", help="Local directory for datasets, checkpoints, samples, reports.")
_AP.add_argument("--profile", default="standard_phase1", choices=["quick_smoke", "standard_phase1", "full_quality"])
_AP.add_argument("--dataset", default="bznsyp", choices=["bznsyp", "ljspeech", "custom_bilingual"])
_AP.add_argument("--vocoder", default="both", choices=["both", "pretrained_vocos", "causal_vocos_lite"])
_AP.add_argument("--models", default="", help="Optional comma-separated subset, e.g. E1_conv_unigru_L4,E2_swa_L4")
_AP.add_argument("--backup-steps", type=int, default=1000)
_AP.add_argument("--backup-minutes", type=float, default=8.0)
_AP.add_argument("--skip-pip", action="store_true")
_AP.add_argument("--selftest", action="store_true", help="Run bit-equality selftest of vectorized hot paths + architecture gates, then exit.")
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
if _ARGS.skip_pip:
    _os.environ["AETHER_SKIP_PIP"] = "1"
SELFTEST = _ARGS.selftest
PROFILE_STEPS = _ARGS.profile_steps


# #############################################################################
# Cell 1: 环境识别与全局配置
# #############################################################################
import os
import sys
import json
import math
import time
import signal
import shutil
import random
import tarfile
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

# ==============================================================================
# 1) 环境自动检测（支持环境变量覆盖，方便自建 4090 CLI 脚本调用）
# ==============================================================================
ENV_MODE = os.environ.get("AETHER_ENV_MODE", "auto").lower()  # "auto" | "colab" | "local" | "kaggle"
if ENV_MODE == "auto":
    if "google.colab" in sys.modules:
        ENV_MODE = "colab"
    elif Path("/kaggle/working").exists():
        ENV_MODE = "kaggle"
    else:
        ENV_MODE = "local"

IS_COLAB = (ENV_MODE == "colab")
IS_LOCAL = (ENV_MODE == "local")
IS_KAGGLE = (ENV_MODE == "kaggle")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    STORAGE_ROOT = Path(os.environ.get("AETHER_ROOT", "/content/drive/MyDrive/AetherTTS_Exp"))
    LOCAL_WORK = Path("/content/aethertts_work")
elif IS_KAGGLE:
    STORAGE_ROOT = Path(os.environ.get("AETHER_ROOT", "/kaggle/working/AetherTTS_Exp"))
    LOCAL_WORK = Path("/kaggle/temp/aethertts_work")
else:
    # 自建 4090 / 本地 Linux：所有文件统一留在本地，不做任何云盘挂载或重复拷贝！
    STORAGE_ROOT = Path(os.environ.get("AETHER_ROOT", "./aethertts_exp")).expanduser().resolve()
    LOCAL_WORK = STORAGE_ROOT

# 兼容别名（方便后续统一引用）
DRIVE_ROOT = STORAGE_ROOT

for p in [
    STORAGE_ROOT / "datasets" / "raw",
    STORAGE_ROOT / "vocoder",
    STORAGE_ROOT / "runs",
    STORAGE_ROOT / "eval_samples",
    STORAGE_ROOT / "reports",
    LOCAL_WORK / "data",
    LOCAL_WORK / "cache",
    LOCAL_WORK / "vocoder",
    LOCAL_WORK / "runs",
    LOCAL_WORK / "eval_samples",
    LOCAL_WORK / "reports",
]:
    p.mkdir(parents=True, exist_ok=True)

# 针对 4090 / Ampere+ GPU 启用 TF32 加速
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# 2) 用户核心配置区（支持环境变量覆写；国内自建 4090 可提前 export HF_ENDPOINT=https://hf-mirror.com）
# ==============================================================================
# 实验档位："quick_smoke" | "standard_phase1" | "full_quality"
EXPERIMENT_PROFILE = os.environ.get("AETHER_PROFILE", "standard_phase1")

# 数据集选择："bznsyp" | "ljspeech" | "custom_bilingual"
DATASET_SOURCE = os.environ.get("AETHER_DATASET", "bznsyp")

# 声码器模式："both" | "pretrained_vocos" | "causal_vocos_lite"
VOCODER_MODE = os.environ.get("AETHER_VOCODER", "both")

# 要训练和评测的模型列表：(实验名, 主干类型, 文本右前瞻音素数 L)
ALL_EXPERIMENTS = [
    ("E0_offline_bitransformer", "offline_transformer", 0),
    ("E1_conv_unigru_L4",        "conv_gru",            4),
    ("E2_swa_L4",                "swa",                 4),
    ("E3_ssm_mamba_L4",          "ssm",                 4),
    ("E4_rwkv_L4",               "rwkv",                4),
]
_filter_models = os.environ.get("AETHER_MODELS", "").strip()
if _filter_models:
    _wanted = {m.strip() for m in _filter_models.split(",") if m.strip()}
    EXPERIMENTS = [e for e in ALL_EXPERIMENTS if e[0] in _wanted]
else:
    EXPERIMENTS = ALL_EXPERIMENTS

# 断点续训与高频备份策略（支持环境变量覆写）
SKIP_IF_DONE = True                                                        # 仅当存在 .done 完结标记时才跳过；未完结则自动从 latest.pt 精确续训
BACKUP_EVERY_STEPS = int(os.environ.get("AETHER_BACKUP_STEPS", "1000"))    # 声学模型每 1000 步保存并同步 latest.pt（防 Colab T4/L4 断连）
VOCODER_BACKUP_EVERY_STEPS = int(os.environ.get("AETHER_VOC_BACKUP_STEPS", "1500"))  # 声码器每 1500 步保存并同步 latest.pt
BACKUP_EVERY_MINUTES = float(os.environ.get("AETHER_BACKUP_MINUTES", "8.0"))         # 时间兜底：每满 8 分钟强制触发一次断点备份
SAVE_MILESTONE_EVERY = 5000                                                # 每 5000 步保留一份不可覆盖的里程碑权重快照
SYNC_FEATURES_TAR_TO_DRIVE = IS_COLAB                                      # Colab 下把预处理好的特征分片打包备份到云盘，重连秒恢复；自建本地就地缓存
CACHE_RAW_TO_DRIVE = (os.environ.get("AETHER_CACHE_RAW_DRIVE", "0") == "1") if not IS_LOCAL else True  # Colab 下默认不上传/回读 3GB+ 原始包到 DriveFS，防 FUSE 卡死断连
AUTO_UNASSIGN_COLAB = False                                                # Colab Pro 后台跑完后是否自动释放运行时（默认 False）

# 根据显存自动适配 batch_size（T4 15.8GB -> 16 防 OOM；L4/4090 24GB -> 32 提速）
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
CPU_CORES = os.cpu_count() or 2
NUM_WORKERS = min(6, max(2, CPU_CORES // 2)) if IS_LOCAL else 2

PROFILE_PRESETS = {
    "quick_smoke": {
        "max_train_hours": 2.0,
        "val_utts": 40,
        "vocoder_stage1_steps": 2000,
        "vocoder_stage2_steps": 1000,
        "acoustic_steps": 3000,
        "batch_size": 16,
        "eval_max_utts": 12,
    },
    "standard_phase1": {
        "max_train_hours": 8.0,
        "val_utts": 80,
        "vocoder_stage1_steps": 12000,
        "vocoder_stage2_steps": 6000,
        "acoustic_steps": 15000,
        "batch_size": 32 if GPU_MEM_GB >= 22 else (24 if GPU_MEM_GB >= 18 else 16),
        "eval_max_utts": 25,
    },
    "full_quality": {
        "max_train_hours": 20.0,
        "val_utts": 100,
        "vocoder_stage1_steps": 35000,
        "vocoder_stage2_steps": 15000,
        "acoustic_steps": 40000,
        "batch_size": 32 if GPU_MEM_GB >= 22 else 16,
        "eval_max_utts": 50,
    },
}

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
    "stream_text_chunk": 3,
    "stream_mel_chunk": 16,
    "lr_acoustic": 4e-4,
    "lr_vocoder": 2e-4,
    "env_mode": ENV_MODE,
    "profile": EXPERIMENT_PROFILE,
    "dataset_source": DATASET_SOURCE,
    "vocoder_mode": VOCODER_MODE,
    "num_workers": NUM_WORKERS,
    **PROFILE_PRESETS[EXPERIMENT_PROFILE],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

print("=" * 72)
print(f"Environment   : {ENV_MODE.upper()} ({'纯本地模式，无云盘拷贝' if IS_LOCAL else '云盘同步备份模式'})")
print(f"Device        : {DEVICE}" + (f" | {torch.cuda.get_device_name(0)} ({GPU_MEM_GB:.1f} GB)" if torch.cuda.is_available() else ""))
print(f"CPU Cores     : {CPU_CORES} (DataLoader workers={NUM_WORKERS})")
print(f"Storage Root  : {STORAGE_ROOT}")
print(f"Local Work    : {LOCAL_WORK}")
if os.environ.get("HF_ENDPOINT"):
    print(f"HF Endpoint   : {os.environ['HF_ENDPOINT']}")
print(f"Profile       : {EXPERIMENT_PROFILE} | max_train_hours={CFG['max_train_hours']}h | batch={CFG['batch_size']} | steps={CFG['acoustic_steps']}")
print(f"Backup Cadence: every {BACKUP_EVERY_STEPS} steps or {BACKUP_EVERY_MINUTES:.1f} min (+ milestone every {SAVE_MILESTONE_EVERY} steps + SIGTERM/Ctrl+C emergency save)")
print(f"Models        : {[x[0] for x in EXPERIMENTS]}")
print("=" * 72)

# #############################################################################
# Cell 2: 依赖安装
# #############################################################################
import importlib.metadata
import importlib.util
import subprocess

AUTO_INSTALL_DEPS = os.environ.get("AETHER_SKIP_PIP", "0") != "1"

# 保护 Colab / 4090 环境中已预装的基础包版本，避免 pip 升级破坏 CUDA PyTorch / pandas / huggingface-hub
PROTECTED_BASE_PKGS = [
    "torch", "torchaudio", "torchvision", "numpy", "pandas",
    "scipy", "matplotlib", "huggingface-hub", "decorator",
]

CORE_PKGS = [
    ("pypinyin", "pypinyin"),
    ("g2p_en", "g2p-en"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("scipy", "scipy"),
    ("einops", "einops"),
    ("tqdm", "tqdm"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("huggingface_hub", "huggingface-hub"),
    ("jiwer", "jiwer"),
]

OPTIONAL_EVAL_PKGS = [
    ("vocos", "vocos"),
    ("faster_whisper", "faster-whisper"),
]


def _pip_install(pkgs, constraints_file: Path, required: bool = True):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints_file), *pkgs]
    ret = subprocess.call(cmd)
    if ret != 0:
        # 若约束过严导致解析失败，回退为普通安装（保持 --upgrade-strategy only-if-needed）
        fallback_cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *pkgs]
        if required:
            subprocess.check_call(fallback_cmd)
        else:
            ret2 = subprocess.call(fallback_cmd)
            if ret2 != 0:
                print(f"[Pip Warn] 可选包安装失败（不影响主训练，评测时自动降级跳过）: {pkgs}")


if AUTO_INSTALL_DEPS:
    constraints_lines = []
    for pkg in PROTECTED_BASE_PKGS:
        try:
            ver = importlib.metadata.version(pkg)
            constraints_lines.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            pass
    constraints_file = LOCAL_WORK / "pip_constraints.txt"
    constraints_file.write_text("\n".join(constraints_lines) + "\n", encoding="utf-8")

    missing_core = [pip_name for mod_name, pip_name in CORE_PKGS if importlib.util.find_spec(mod_name) is None]
    if missing_core:
        print(f"[Pip] 正在安装缺失核心依赖: {missing_core}")
        _pip_install(missing_core, constraints_file, required=True)
    else:
        print("[Pip] 核心训练依赖均已就绪，跳过安装。")

    missing_opt = [pip_name for mod_name, pip_name in OPTIONAL_EVAL_PKGS if importlib.util.find_spec(mod_name) is None]
    if missing_opt:
        print(f"[Pip] 正在安装可选评测/对比依赖: {missing_opt}")
        _pip_install(missing_opt, constraints_file, required=False)

    # resemblyzer 依赖老版 webrtcvad（Python 3.12 源码编译慢/易报错），改用预编译 webrtcvad-wheels + --no-deps 秒装
    if importlib.util.find_spec("resemblyzer") is None:
        print("[Pip] 正在安装声纹评测依赖: webrtcvad-wheels + resemblyzer（免源码编译）...")
        r1 = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "webrtcvad-wheels"])
        r2 = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "resemblyzer"])
        if r1 != 0 or r2 != 0:
            print("[Pip Warn] resemblyzer 安装未成功，Cell 11 将自动跳过 SIM 声纹指标，不影响训练与其余评测。")

    print("[Pip] 依赖检查与安装已完成！（提示：Colab 下若上方出现 cudf/moviepy/google-colab 等镜像自带的红色冲突提示，属于正常现象，可直接运行后续 Cell。）")
else:
    print("[Pip] 已设置 AETHER_SKIP_PIP=1，跳过自动安装。")

# #############################################################################
# Cell 3: 基础设施（Mel/F0/前端）
# #############################################################################
import gc
import re
import ctypes
import subprocess
import unicodedata
import numpy as np
import soundfile as sf
import librosa
from scipy import signal as scipy_signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pypinyin import pinyin, Style

try:
    from g2p_en import G2p
    import nltk
    for res in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    _EN_G2P = G2p()
except Exception as e:
    print(f"[Warn] g2p_en unavailable ({e}), falling back to character-level EN G2P.")
    _EN_G2P = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG["seed"])


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def trim_process_memory():
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def decode_wav_tensor(w: torch.Tensor) -> torch.Tensor:
    return (w.float() / 32767.0) if w.dtype == torch.int16 else w.float()


def atomic_torch_save(obj, path: Path):
    """原子写盘，防止断电/断连写坏 checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sync_file_to_storage(local_path: Path, storage_path: Path):
    """Colab 下原子同步到云盘；自建 4090 本地模式下若源与目标相同则零开销直接跳过"""
    if _same_path(local_path, storage_path):
        return
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = storage_path.with_suffix(storage_path.suffix + ".tmp")
    shutil.copy2(local_path, tmp_path)
    os.replace(tmp_path, storage_path)


def copy_from_storage_to_local(storage_path: Path, local_path: Path, chunk_mb: int = 8) -> Path:
    """将云盘大文件按 8MB 顺序分块复制到本地 SSD，避免在 DriveFS FUSE 上直接小块流式读取导致断连"""
    if _same_path(storage_path, local_path):
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    total = storage_path.stat().st_size
    copied = 0
    last_log = 0
    buf_sz = chunk_mb * 1024 * 1024
    with open(storage_path, "rb") as fsrc, open(tmp_path, "wb") as fdst:
        while True:
            buf = fsrc.read(buf_sz)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            if total >= 200 * 1024 * 1024 and (copied - last_log) >= 256 * 1024 * 1024:
                last_log = copied
                print(f"  [Stage] {storage_path.name}: {copied / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    os.replace(tmp_path, local_path)
    return local_path


def sync_dir_to_storage(local_dir: Path, storage_dir: Path):
    if _same_path(local_dir, storage_dir):
        return
    storage_dir.mkdir(parents=True, exist_ok=True)
    for src in local_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(local_dir)
            sync_file_to_storage(src, storage_dir / rel)


def atomic_write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_progress_status(stage: str, detail: dict):
    """在 STORAGE_ROOT/progress.json 实时记录运行进度，云盘/本地随时可一眼查看"""
    prog_local = LOCAL_WORK / "progress.json"
    prog_storage = STORAGE_ROOT / "progress.json"
    try:
        cur = json.loads(prog_storage.read_text(encoding="utf-8")) if prog_storage.exists() else {}
    except Exception:
        cur = {}
    cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cur["env_mode"] = ENV_MODE
    cur["profile"] = CFG["profile"]
    cur["dataset"] = CFG["dataset_source"]
    cur.setdefault("stages", {})[stage] = detail
    txt = json.dumps(cur, ensure_ascii=False, indent=2)
    atomic_write_text(txt, prog_local)
    sync_file_to_storage(prog_local, prog_storage)


_STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[Signal] 收到 SIGTERM 停止信号，当前 step 完成后将立即执行紧急断点备份并安全退出！", flush=True)


try:
    signal.signal(signal.SIGTERM, _handle_sigterm)
except Exception:
    pass



# ------------------------------------------------------------------------------
# 24 kHz 100-bin Log-Mel 提取器（与 charactr/vocos-mel-24khz 100% 参数对齐）
# ------------------------------------------------------------------------------
class VocosCompatibleMel(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True
        mel = self.mel_spec(wav)
        log_mel = torch.log(torch.clamp(mel, min=1e-7))
        return log_mel.squeeze(0) if squeeze else log_mel


MEL_EXTRACTOR = VocosCompatibleMel(
    sample_rate=CFG["sample_rate"],
    n_fft=CFG["n_fft"],
    hop_length=CFG["hop_length"],
    n_mels=CFG["n_mels"],
)


def load_and_normalize_wav(path: Path, target_sr: int = 24000) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        if sr > target_sr and sr % target_sr == 0:
            wav = scipy_signal.resample_poly(wav, 1, sr // target_sr).astype(np.float32, copy=False)
        elif target_sr > sr and target_sr % sr == 0:
            wav = scipy_signal.resample_poly(wav, target_sr // sr, 1).astype(np.float32, copy=False)
        else:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav_trimmed, idx = librosa.effects.trim(wav, top_db=35, frame_length=1024, hop_length=256)
    pad = int(0.06 * target_sr)
    start = max(0, idx[0] - pad)
    end = min(len(wav), idx[1] + pad)
    wav = wav[start:end]
    rms = np.sqrt(np.mean(wav ** 2) + 1e-8)
    wav = wav * (0.10 / max(rms, 1e-4))
    peak = np.max(np.abs(wav)) + 1e-8
    if peak > 0.92:
        wav = wav * (0.92 / peak)
    return wav.astype(np.float32)


def extract_f0_and_energy(wav: np.ndarray, sr: int = 24000, hop_length: int = 256, n_frames: int = None):
    f0 = librosa.yin(
        wav, fmin=65.0, fmax=600.0, sr=sr, frame_length=1024, hop_length=hop_length, center=True
    ).astype(np.float32)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop_length, center=True)[0].astype(np.float32)
    if n_frames is not None:
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)), mode="edge")
            rms = np.pad(rms, (0, n_frames - len(rms)), mode="edge")
        else:
            f0 = f0[:n_frames]
            rms = rms[:n_frames]
    uv = (rms > 0.008) & (f0 > 70.0) & (f0 < 580.0)
    log_f0 = np.zeros_like(f0, dtype=np.float32)
    if uv.any():
        voiced_idx = np.where(uv)[0]
        voiced_vals = np.log(f0[voiced_idx])
        log_f0 = np.interp(np.arange(len(f0)), voiced_idx, voiced_vals).astype(np.float32)
    log_energy = np.log(np.clip(rms, 1e-5, None)).astype(np.float32)
    return log_f0, log_energy, uv.astype(np.float32)


# ------------------------------------------------------------------------------
# 统一中英混读音素前端 (ZH 声母/带调韵母 + EN ARPAbet + 韵律/标点 + LangID)
# ------------------------------------------------------------------------------
ZH_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
]
ZH_FINALS_BASE = [
    "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ve", "er",
    "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong",
    "ia", "iao", "ian", "iang", "iong", "ua", "uo", "uai", "uan", "uang", "ue", "van",
]
EN_ARPABET = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sp>", ",", ".", "?", "!", ";", ":", "#1", "#2", "#3"]


class BilingualPhonemeFrontend:
    def __init__(self):
        vocab = list(SPECIAL_TOKENS)
        vocab += [f"zh_{ini}" for ini in ZH_INITIALS]
        for fin in ZH_FINALS_BASE:
            for tone in ["1", "2", "3", "4", "5"]:
                vocab.append(f"zh_{fin}{tone}")
        vocab += [f"en_{ph}" for ph in EN_ARPABET]
        self.vocab = vocab
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self.pad_id = self.tok2id["<pad>"]
        self.bos_id = self.tok2id["<bos>"]
        self.eos_id = self.tok2id["<eos>"]
        self.unk_id = self.tok2id["<unk>"]
        self.sp_id = self.tok2id["<sp>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _num_to_zh(num_str: str) -> str:
        digits = "零一二三四五六七八九"
        if "." in num_str:
            int_p, dec_p = num_str.split(".", 1)
            return BilingualPhonemeFrontend._num_to_zh(int_p) + "点" + "".join(digits[int(c)] for c in dec_p if c.isdigit())
        n = int(num_str)
        if len(num_str) == 4 and 1900 <= n <= 2099:
            return "".join(digits[int(c)] for c in num_str)
        if n == 0:
            return "零"
        if n < 10:
            return digits[n]
        if n < 20:
            return "十" + ("" if n == 10 else digits[n % 10])
        if n < 100:
            return digits[n // 10] + "十" + ("" if n % 10 == 0 else digits[n % 10])
        return "".join(digits[int(c)] for c in num_str)

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).strip()
        punc_map = {"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "、": ",", "—": ",", "…": "."}
        for k, v in punc_map.items():
            text = text.replace(k, v)
        has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        if has_zh:
            text = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: "百分之" + self._num_to_zh(m.group(1)), text)
            text = re.sub(r"\d+(?:\.\d+)?", lambda m: self._num_to_zh(m.group(0)), text)
        return text

    @staticmethod
    def _split_zh_syllable(syl: str):
        syl = syl.lower().replace("ü", "v").replace("u:", "v")
        tone = syl[-1] if syl and syl[-1] in "12345" else "5"
        body = syl[:-1] if syl and syl[-1] in "12345" else syl
        if not body:
            return []
        ini = ""
        for cand in sorted(ZH_INITIALS, key=len, reverse=True):
            if body.startswith(cand) and len(body) > len(cand):
                ini = cand
                body = body[len(cand):]
                break
        if len(body) > 1 and body.endswith("r") and body != "er":
            body = body[:-1]
        if body == "u" and ini in ("j", "q", "x", "y"):
            body = "v"
        elif body == "ue" and ini in ("j", "q", "x", "y"):
            body = "ve"
        elif body == "uan" and ini in ("j", "q", "x", "y"):
            body = "van"
        elif body == "un" and ini in ("j", "q", "x", "y"):
            body = "vn"
        toks = []
        if ini:
            toks.append(f"zh_{ini}")
        toks.append(f"zh_{body}{tone}")
        return toks

    def _zh_span_to_phonemes(self, span: str):
        pys = [p[0] for p in pinyin(span, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore") if p]
        for i in range(len(pys) - 1):
            if pys[i].endswith("3") and pys[i + 1].endswith("3"):
                pys[i] = pys[i][:-1] + "2"
        out = []
        for s in pys:
            out.extend(self._split_zh_syllable(s))
        return out

    def _en_span_to_phonemes(self, span: str):
        if _EN_G2P is not None:
            raw = _EN_G2P(span)
            out = []
            for p in raw:
                if p == " ":
                    out.append("<sp>")
                elif f"en_{p}" in self.tok2id:
                    out.append(f"en_{p}")
            return out
        return ["<sp>"]

    def text_to_ids(self, text: str, prosody_pinyin: list = None):
        tokens = ["<bos>"]
        langs = [0]
        if prosody_pinyin is not None:
            for item in prosody_pinyin:
                if item in self.tok2id:
                    tokens.append(item); langs.append(0)
                else:
                    for ph in self._split_zh_syllable(item):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(1)
        else:
            norm = self.normalize_text(text)
            pattern = re.compile(r"([\u4e00-\u9fff]+|[A-Za-z][A-Za-z'\-]*|#1|#2|#3|[,\.\?!;:])")
            prev_lang = 0
            for m in pattern.finditer(norm):
                seg = m.group(0)
                if re.match(r"^[\u4e00-\u9fff]+$", seg):
                    if prev_lang == 2:
                        tokens.append("<sp>"); langs.append(0)
                    for ph in self._zh_span_to_phonemes(seg):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(1)
                    prev_lang = 1
                elif re.match(r"^[A-Za-z]", seg):
                    if prev_lang in (1, 2):
                        tokens.append("<sp>"); langs.append(0)
                    for ph in self._en_span_to_phonemes(seg):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(2 if ph.startswith("en_") else 0)
                    prev_lang = 2
                else:
                    if seg in self.tok2id:
                        tokens.append(seg); langs.append(0)
                    prev_lang = 0
        tokens.append("<eos>")
        langs.append(0)
        return [self.tok2id.get(t, self.unk_id) for t in tokens], langs


FRONTEND = BilingualPhonemeFrontend()
print(f"Unified Bilingual Vocab Size: {FRONTEND.vocab_size}")
for demo in ["你好，请问今天北京的天气怎么样？", "Hello world, how is your API latency today?", "帮我把 Wi-Fi 切换到 5G，谢谢！"]:
    pids, lids = FRONTEND.text_to_ids(demo)
    print(f"  {demo} -> {len(pids)} phonemes")

# #############################################################################
# Cell 5: 模型（含向量化对齐热路径）
# #############################################################################
# ==============================================================================
# 5.1 集中式有限右前瞻层 & 严格因果 ConvNeXt 块（100% 数值等价流式缓存）
# ==============================================================================
class CentralizedLookaheadBlock(nn.Module):
    def __init__(self, d_model: int, k_left: int = 3, lookahead: int = 4, mult: int = 2):
        super().__init__()
        self.d_model = d_model
        self.k_left = k_left
        self.lookahead = lookahead
        self.kernel_size = k_left + 1 + lookahead
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=self.kernel_size, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * mult)
        self.pw2 = nn.Linear(d_model * mult, d_model)

    def _apply_body(self, conv_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        h = self.norm(conv_out.transpose(1, 2))
        h = self.pw2(F.gelu(self.pw1(h)))
        return residual + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x.transpose(1, 2), (self.k_left, self.lookahead))
        return self._apply_body(self.dwconv(x_pad), x)

    def init_state(self, batch_size: int, device, dtype):
        return torch.zeros(batch_size, self.k_left, self.d_model, device=device, dtype=dtype)

    def forward_step(self, x_new: torch.Tensor, state: torch.Tensor, is_final: bool = False):
        buf = torch.cat([state, x_new], dim=1) if x_new.shape[1] > 0 else state
        if is_final and self.lookahead > 0:
            zeros_r = torch.zeros(buf.shape[0], self.lookahead, self.d_model, device=buf.device, dtype=buf.dtype)
            buf = torch.cat([buf, zeros_r], dim=1)
        if buf.shape[1] < self.kernel_size:
            return torch.zeros(buf.shape[0], 0, self.d_model, device=buf.device, dtype=buf.dtype), buf
        n_emit = buf.shape[1] - self.kernel_size + 1
        conv_out = self.dwconv(buf.transpose(1, 2))
        residual = buf[:, self.k_left : self.k_left + n_emit, :]
        return self._apply_body(conv_out, residual), buf[:, n_emit:, :]


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, groups: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_left = kernel_size - 1
        self.in_ch = in_ch
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad_left, 0)))

    def init_state(self, batch_size: int, device, dtype):
        return torch.zeros(batch_size, self.in_ch, self.pad_left, device=device, dtype=dtype)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor):
        if x.shape[-1] == 0:
            return torch.zeros(x.shape[0], self.conv.out_channels, 0, device=x.device, dtype=x.dtype), state
        buf = torch.cat([state, x], dim=-1)
        out = self.conv(buf)
        new_state = buf[:, :, -self.pad_left:] if self.pad_left > 0 else state
        return out, new_state


class CausalConvNeXtBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, mult: int = 3):
        super().__init__()
        self.dwconv = CausalConv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * mult)
        self.pw2 = nn.Linear(d_model * mult, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + h

    def init_state(self, batch_size: int, device, dtype):
        return self.dwconv.init_state(batch_size, device, dtype)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor):
        if x.shape[1] == 0:
            return x, state
        h, new_state = self.dwconv.forward_step(x.transpose(1, 2), state)
        h = self.pw2(F.gelu(self.pw1(self.norm(h.transpose(1, 2)))))
        return x + h, new_state


# ==============================================================================
# 5.2 五大序列主干（E0 离线双向 / E1 Conv+UniGRU / E2 SWA / E3 SSM / E4 RWKV）
# ==============================================================================
def apply_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    dh = x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    freqs = torch.outer(pos.to(torch.float32), inv_freq)
    cos = freqs.cos()[None, None, :, :].to(x.dtype)
    sin = freqs.sin()[None, None, :, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class OfflineBiTransformerBackbone(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, d_ff: int = 896):
        super().__init__()
        self.is_streaming = False
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.pos_emb = nn.Embedding(512, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).clamp(max=511)
        h = x + self.pos_emb(pos)[None, :, :]
        pad_mask = (~mask) if mask is not None else None
        return self.encoder(h, src_key_padding_mask=pad_mask)


class ConvUniGRUBackbone(nn.Module):
    def __init__(self, d_model: int = 256, k_left: int = 3, lookahead: int = 4, n_conv: int = 4):
        super().__init__()
        self.is_streaming = True
        self.lookahead_layer = CentralizedLookaheadBlock(d_model, k_left=k_left, lookahead=lookahead, mult=2)
        self.convs = nn.ModuleList([CausalConvNeXtBlock(d_model, kernel_size=5, mult=3) for _ in range(n_conv)])
        self.gru_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, bidirectional=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_ffn = nn.Sequential(nn.Linear(d_model, d_model * 3), nn.GELU(), nn.Linear(d_model * 3, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.lookahead_layer(x)
        for blk in self.convs:
            h = blk(h)
        gru_out, _ = self.gru(self.gru_norm(h))
        h = h + gru_out
        return h + self.out_ffn(self.out_norm(h))

    def init_state(self, batch_size: int, device, dtype):
        return {
            "la": self.lookahead_layer.init_state(batch_size, device, dtype),
            "convs": [blk.init_state(batch_size, device, dtype) for blk in self.convs],
            "gru": torch.zeros(2, batch_size, self.gru.hidden_size, device=device, dtype=dtype),
        }

    def forward_step(self, x_new: torch.Tensor, state: dict, is_final: bool = False):
        h, la_st = self.lookahead_layer.forward_step(x_new, state["la"], is_final=is_final)
        if h.shape[1] == 0:
            return h, {"la": la_st, "convs": state["convs"], "gru": state["gru"]}
        new_convs = []
        for blk, cst in zip(self.convs, state["convs"]):
            h, nc = blk.forward_step(h, cst)
            new_convs.append(nc)
        gru_out, new_gru = self.gru(self.gru_norm(h), state["gru"])
        h = h + gru_out
        h = h + self.out_ffn(self.out_norm(h))
        return h, {"la": la_st, "convs": new_convs, "gru": new_gru}


class SWAConvLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, window_size: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.conv_ffn = CausalConvNeXtBlock(d_model, kernel_size=5, mult=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        pos = torch.arange(T, device=x.device)
        q, k = apply_rope(q, pos), apply_rope(k, pos)
        diff = pos[:, None] - pos[None, :]
        allow = (diff >= 0) & (diff < self.window_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allow[None, None, :, :], float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, T, D)
        return self.conv_ffn(x + self.proj(out))

    def init_state(self, batch_size: int, device, dtype):
        return {
            "k": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "pos": 0,
            "conv": self.conv_ffn.init_state(batch_size, device, dtype),
        }

    def forward_step(self, x: torch.Tensor, state: dict):
        B, C, D = x.shape
        if C == 0:
            return x, state
        qkv = self.qkv(self.norm1(x)).view(B, C, 3, self.n_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        pos_q = torch.arange(state["pos"], state["pos"] + C, device=x.device)
        q, k = apply_rope(q, pos_q), apply_rope(k, pos_q)
        k_all = torch.cat([state["k"], k], dim=2)
        v_all = torch.cat([state["v"], v], dim=2)
        k_len = k_all.shape[2]
        pos_k = torch.arange(state["pos"] + C - k_len, state["pos"] + C, device=x.device)
        diff = pos_q[:, None] - pos_k[None, :]
        allow = (diff >= 0) & (diff < self.window_size)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allow[None, None, :, :], float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).contiguous().view(B, C, D)
        x, new_conv = self.conv_ffn.forward_step(x + self.proj(out), state["conv"])
        keep = self.window_size - 1
        return x, {
            "k": k_all[:, :, -keep:, :] if keep > 0 else k_all[:, :, :0, :],
            "v": v_all[:, :, -keep:, :] if keep > 0 else v_all[:, :, :0, :],
            "pos": state["pos"] + C,
            "conv": new_conv,
        }


class SWABackbone(nn.Module):
    def __init__(self, d_model: int = 256, k_left: int = 3, lookahead: int = 4, n_layers: int = 5, n_heads: int = 4, window_size: int = 48):
        super().__init__()
        self.is_streaming = True
        self.lookahead_layer = CentralizedLookaheadBlock(d_model, k_left=k_left, lookahead=lookahead, mult=2)
        self.layers = nn.ModuleList([SWAConvLayer(d_model, n_heads=n_heads, window_size=window_size) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.lookahead_layer(x)
        for l in self.layers:
            h = l(h)
        return h

    def init_state(self, batch_size: int, device, dtype):
        return {"la": self.lookahead_layer.init_state(batch_size, device, dtype), "layers": [l.init_state(batch_size, device, dtype) for l in self.layers]}

    def forward_step(self, x_new: torch.Tensor, state: dict, is_final: bool = False):
        h, la_st = self.lookahead_layer.forward_step(x_new, state["la"], is_final=is_final)
        if h.shape[1] == 0:
            return h, {"la": la_st, "layers": state["layers"]}
        new_layers = []
        for l, st in zip(self.layers, state["layers"]):
            h, nst = l.forward_step(h, st)
            new_layers.append(nst)
        return h, {"la": la_st, "layers": new_layers}


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model: int = 256, expand: int = 2, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(16, d_model // 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = CausalConv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _scan(self, x_silu: torch.Tensor, h_init: torch.Tensor):
        B_sz, T, _ = x_silu.shape
        dt_raw, B_mat, C_mat = torch.split(self.x_proj(x_silu), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)).float()
        A = -torch.exp(self.A_log.float())
        B_f, C_f, x_f = B_mat.float(), C_mat.float(), x_silu.float()
        h = h_init.float()
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA_t = torch.exp(dt_t * A.unsqueeze(0))
            dBx_t = (dt_t * x_f[:, t, :].unsqueeze(-1)) * B_f[:, t, :].unsqueeze(1)
            h = dA_t * h + dBx_t
            ys.append(torch.sum(h * C_f[:, t, :].unsqueeze(1), dim=-1) + self.D_skip.float() * x_f[:, t, :])
        return torch.stack(ys, dim=1).to(x_silu.dtype), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_b, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x_c = F.silu(self.conv(x_b.transpose(1, 2)).transpose(1, 2))
        h0 = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        y, _ = self._scan(x_c, h0)
        return x + self.out_proj(y * F.silu(z))

    def init_state(self, batch_size: int, device, dtype):
        return {"conv": self.conv.init_state(batch_size, device, dtype), "h": torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=torch.float32)}

    def forward_step(self, x: torch.Tensor, state: dict):
        if x.shape[1] == 0:
            return x, state
        x_b, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x_c, new_conv = self.conv.forward_step(x_b.transpose(1, 2), state["conv"])
        y, new_h = self._scan(F.silu(x_c.transpose(1, 2)), state["h"])
        return x + self.out_proj(y * F.silu(z)), {"conv": new_conv, "h": new_h}


class SelectiveSSMBackbone(nn.Module):
    def __init__(self, d_model: int = 256, k_left: int = 3, lookahead: int = 4, n_layers: int = 5):
        super().__init__()
        self.is_streaming = True
        self.lookahead_layer = CentralizedLookaheadBlock(d_model, k_left=k_left, lookahead=lookahead, mult=2)
        self.layers = nn.ModuleList([SelectiveSSMBlock(d_model, expand=2, d_state=16, d_conv=4) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.lookahead_layer(x)
        for l in self.layers:
            h = l(h)
        return h

    def init_state(self, batch_size: int, device, dtype):
        return {"la": self.lookahead_layer.init_state(batch_size, device, dtype), "layers": [l.init_state(batch_size, device, dtype) for l in self.layers]}

    def forward_step(self, x_new: torch.Tensor, state: dict, is_final: bool = False):
        h, la_st = self.lookahead_layer.forward_step(x_new, state["la"], is_final=is_final)
        if h.shape[1] == 0:
            return h, {"la": la_st, "layers": state["layers"]}
        new_layers = []
        for l, st in zip(self.layers, state["layers"]):
            h, nst = l.forward_step(h, st)
            new_layers.append(nst)
        return h, {"la": la_st, "layers": new_layers}


class RWKVBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, ffn_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.mu_r = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mu_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mu_v = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mu_w = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mu_g = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.w_base = nn.Parameter(torch.full((n_heads, self.head_dim), -1.0))
        self.w_lora1 = nn.Linear(d_model, 32, bias=False)
        self.w_lora2 = nn.Linear(32, d_model, bias=True)
        self.u_bonus = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.head_norm = nn.GroupNorm(n_heads, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.mu_ck = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.mu_cr = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.ck = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.cv = nn.Linear(d_model * ffn_mult, d_model, bias=False)
        self.cr = nn.Linear(d_model, d_model, bias=False)

    def _run(self, x: torch.Tensor, last_t: torch.Tensor, wkv_s: torch.Tensor, last_c: torch.Tensor):
        B, T, D = x.shape
        xn = self.norm1(x)
        prev_t = torch.cat([last_t, xn[:, :-1, :]], dim=1)
        xr = xn * self.mu_r + prev_t * (1.0 - self.mu_r)
        xk = xn * self.mu_k + prev_t * (1.0 - self.mu_k)
        xv = xn * self.mu_v + prev_t * (1.0 - self.mu_v)
        xw = xn * self.mu_w + prev_t * (1.0 - self.mu_w)
        xg = xn * self.mu_g + prev_t * (1.0 - self.mu_g)

        r = self.receptance(xr).view(B, T, self.n_heads, self.head_dim).float()
        k = self.key(xk).view(B, T, self.n_heads, self.head_dim).float()
        v = self.value(xv).view(B, T, self.n_heads, self.head_dim).float()
        w_dyn = self.w_lora2(torch.tanh(self.w_lora1(xw))).view(B, T, self.n_heads, self.head_dim).float()
        decay = torch.exp(-torch.exp(self.w_base[None, None, :, :].float() + w_dyn).clamp(max=10.0))
        g = F.silu(self.gate(xg))

        S = wkv_s.float()
        u = self.u_bonus[None, :, :].float()
        ys = []
        for t in range(T):
            rt, kt, vt, wt = r[:, t], k[:, t], v[:, t], decay[:, t]
            yt = torch.einsum("bhd,bhde->bhe", rt, S) + (rt * u * kt).sum(dim=-1, keepdim=True) * vt
            S = wt.unsqueeze(-1) * S + torch.einsum("bhd,bhe->bhde", kt, vt)
            ys.append(yt)
        # 逐时间步独立 GroupNorm（严防跨时间轴统计泄漏，保证流式等价）
        y = torch.stack(ys, dim=1).reshape(B * T, D).to(x.dtype)
        y = self.head_norm(y).reshape(B, T, D)
        x = x + self.out_proj(y * g)
        new_last_t = xn[:, -1:, :]

        xc = self.norm2(x)
        prev_c = torch.cat([last_c, xc[:, :-1, :]], dim=1)
        xck = xc * self.mu_ck + prev_c * (1.0 - self.mu_ck)
        xcr = xc * self.mu_cr + prev_c * (1.0 - self.mu_cr)
        x = x + torch.sigmoid(self.cr(xcr)) * self.cv(F.relu(self.ck(xck)) ** 2)
        return x, new_last_t, S, xc[:, -1:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        z1 = torch.zeros(B, 1, self.d_model, device=x.device, dtype=x.dtype)
        S0 = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        out, _, _, _ = self._run(x, z1, S0, z1)
        return out

    def init_state(self, batch_size: int, device, dtype):
        return {
            "last_t": torch.zeros(batch_size, 1, self.d_model, device=device, dtype=dtype),
            "S": torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32),
            "last_c": torch.zeros(batch_size, 1, self.d_model, device=device, dtype=dtype),
        }

    def forward_step(self, x: torch.Tensor, state: dict):
        if x.shape[1] == 0:
            return x, state
        out, lt, S, lc = self._run(x, state["last_t"], state["S"], state["last_c"])
        return out, {"last_t": lt, "S": S, "last_c": lc}


class RWKVBackbone(nn.Module):
    def __init__(self, d_model: int = 256, k_left: int = 3, lookahead: int = 4, n_layers: int = 5, n_heads: int = 4):
        super().__init__()
        self.is_streaming = True
        self.lookahead_layer = CentralizedLookaheadBlock(d_model, k_left=k_left, lookahead=lookahead, mult=2)
        self.layers = nn.ModuleList([RWKVBlock(d_model, n_heads=n_heads, ffn_mult=2) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.lookahead_layer(x)
        for l in self.layers:
            h = l(h)
        return h

    def init_state(self, batch_size: int, device, dtype):
        return {"la": self.lookahead_layer.init_state(batch_size, device, dtype), "layers": [l.init_state(batch_size, device, dtype) for l in self.layers]}

    def forward_step(self, x_new: torch.Tensor, state: dict, is_final: bool = False):
        h, la_st = self.lookahead_layer.forward_step(x_new, state["la"], is_final=is_final)
        if h.shape[1] == 0:
            return h, {"la": la_st, "layers": state["layers"]}
        new_layers = []
        for l, st in zip(self.layers, state["layers"]):
            h, nst = l.forward_step(h, st)
            new_layers.append(nst)
        return h, {"la": la_st, "layers": new_layers}


# ==============================================================================
# 5.3 内置单调对齐器 (Beta-Binomial 对角先验 + Soft->Hard 直通二值化，防对齐坍缩) & 完整声学模型
# ==============================================================================
@torch.no_grad()
def build_beta_binomial_log_prior_ref(text_lens: torch.Tensor, mel_lens: torch.Tensor, max_t: int, max_f: int, scaling: float = 1.0) -> torch.Tensor:
    B = text_lens.shape[0]
    device = text_lens.device
    log_prior = torch.full((B, max_f, max_t), -1e4, device=device, dtype=torch.float32)
    for b in range(B):
        F_b = int(mel_lens[b].item())
        T_b = int(text_lens[b].item())
        if T_b <= 1:
            log_prior[b, :F_b, :T_b] = 0.0
            continue
        n = float(T_b - 1)
        f = torch.arange(F_b, device=device, dtype=torch.float32)[:, None]
        t = torch.arange(T_b, device=device, dtype=torch.float32)[None, :]
        alpha = scaling * (f + 1.0)
        beta = scaling * (float(F_b) - f)
        log_comb = torch.lgamma(torch.tensor(n + 1.0, device=device)) - torch.lgamma(t + 1.0) - torch.lgamma(n - t + 1.0)
        log_beta_num = torch.lgamma(t + alpha) + torch.lgamma(n - t + beta) - torch.lgamma(n + alpha + beta)
        log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        log_prior[b, :F_b, :T_b] = (log_comb + log_beta_num - log_beta_den).clamp(min=-50.0)
    return log_prior


class AlignmentEncoder(nn.Module):
    def __init__(self, n_mels: int = 100, d_text: int = 256, d_attn: int = 80, scale: float = 8.0):
        super().__init__()
        self.key_proj = nn.Sequential(nn.Conv1d(d_text, d_text, 3, padding=1), nn.ReLU(), nn.Conv1d(d_text, d_attn, 1))
        self.query_proj = nn.Sequential(nn.Conv1d(n_mels, n_mels * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(n_mels * 2, d_attn, 1))
        self.scale = scale

    def forward(self, mel: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, attn_log_prior: torch.Tensor = None):
        # L2 归一化余弦相似度将声学 logits 严格约束在 [-scale, +scale]，永不发散或淹没 Beta-Binomial 对角先验
        keys = F.normalize(self.key_proj(text_emb.transpose(1, 2)).float(), dim=1)
        queries = F.normalize(self.query_proj(mel).float(), dim=1)
        logits = torch.bmm(queries.transpose(1, 2), keys) * self.scale
        if attn_log_prior is not None:
            logits = logits + attn_log_prior.float()
        logits = logits.masked_fill(~text_mask[:, None, :], -1e4)
        return F.log_softmax(logits, dim=-1), torch.softmax(logits, dim=-1)


def forward_sum_loss(attn_logprob: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    B, F_max, _ = attn_logprob.shape
    blank = torch.full((B, F_max, 1), -1e4, device=attn_logprob.device, dtype=torch.float32)
    log_probs = torch.cat([blank, attn_logprob.float()], dim=-1).transpose(0, 1)
    targets = torch.cat([torch.arange(1, tl + 1, device=attn_logprob.device, dtype=torch.long) for tl in text_lens])
    return F.ctc_loss(log_probs, targets, mel_lens, text_lens, blank=0, zero_infinity=True)


@torch.no_grad()
def viterbi_monotonic_alignment_ref(attn_logprob: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor):
    B, F_max, T_max = attn_logprob.shape
    lp_np = attn_logprob.detach().cpu().float().numpy()
    durations = np.zeros((B, T_max), dtype=np.int64)
    attn_hard = np.zeros((B, F_max, T_max), dtype=np.float32)
    for b in range(B):
        F_b, T_b = int(mel_lens[b].item()), int(text_lens[b].item())
        val = lp_np[b, :F_b, :T_b]
        dp = np.full((F_b, T_b), -1e9, dtype=np.float32)
        dp[0, 0] = val[0, 0]
        for f in range(1, F_b):
            dp[f, 0] = dp[f - 1, 0] + val[f, 0]
            t_max = min(f + 1, T_b)
            dp[f, 1:t_max] = np.maximum(dp[f - 1, 1:t_max], dp[f - 1, 0:t_max - 1]) + val[f, 1:t_max]
        t = T_b - 1
        for f in range(F_b - 1, -1, -1):
            durations[b, t] += 1
            attn_hard[b, f, t] = 1.0
            if f > 0 and t > 0 and (dp[f - 1, t - 1] > dp[f - 1, t] or f <= t):
                t -= 1
    return torch.from_numpy(attn_hard).to(attn_logprob.device), torch.from_numpy(durations).to(attn_logprob.device)




# ==============================================================================
# 5.3b 向量化热路径（与上方 *_ref 参考实现逐位等价；--selftest 随时可验证）
# ==============================================================================
@torch.no_grad()
def build_beta_binomial_log_prior(text_lens: torch.Tensor, mel_lens: torch.Tensor, max_t: int, max_f: int, scaling: float = 1.0) -> torch.Tensor:
    """批向量化 Beta-Binomial 对角先验：一次广播计算整个 (B, F, T)，消除逐样本 Python 循环。

    与 build_beta_binomial_log_prior_ref 逐位一致（相同元素级运算、相同 clamp/-1e4 填充语义）。
    """
    B = text_lens.shape[0]
    device = text_lens.device
    f_col = torch.arange(max_f, device=device, dtype=torch.float32).view(1, max_f, 1)
    t_row = torch.arange(max_t, device=device, dtype=torch.float32).view(1, 1, max_t)
    n = (text_lens.to(torch.float32) - 1.0).view(B, 1, 1)
    f_b = mel_lens.to(torch.float32).view(B, 1, 1)
    alpha = scaling * (f_col + 1.0)                                    # (1,F,1)
    beta = scaling * (f_b - f_col)                                     # (B,F,1)
    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(t_row + 1.0) - torch.lgamma(n - t_row + 1.0)          # (B,1,T)
    log_beta_num = torch.lgamma(t_row + alpha) + torch.lgamma(n - t_row + beta) - torch.lgamma(n + alpha + beta)  # (B,F,T)
    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)                  # (B,F,1)
    dense = (log_comb + log_beta_num - log_beta_den).clamp_(min=-50.0)
    valid = (f_col < f_b) & (t_row < text_lens.to(torch.float32).view(B, 1, 1))      # (B,F,T)
    single_tok = (text_lens <= 1).view(B, 1, 1)
    out = torch.full((B, max_f, max_t), -1e4, device=device, dtype=torch.float32)
    out = torch.where(valid & ~single_tok, dense, out)
    out = torch.where(valid & single_tok, torch.zeros_like(out), out)
    return out


@torch.no_grad()
def viterbi_monotonic_alignment(attn_logprob: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor):
    """批向量化单调 Viterbi 回溯：仅在帧轴 f 上保留 Python 循环，批与 token 全程向量化。

    与 viterbi_monotonic_alignment_ref 逐位一致（相同 DP 递推、相同回溯判据）。
    """
    B, F_max, T_max = attn_logprob.shape
    lp = attn_logprob.detach().cpu().float().numpy()
    mel_np = mel_lens.detach().cpu().numpy().astype(np.int64)
    txt_np = text_lens.detach().cpu().numpy().astype(np.int64)

    dp = np.full((B, F_max, T_max), -1e9, dtype=np.float32)
    if F_max > 0:
        dp[:, 0, 0] = lp[:, 0, 0]
    for f in range(1, F_max):
        np.add(dp[:, f - 1, 0], lp[:, f, 0], out=dp[:, f, 0])
        t_hi = min(f + 1, T_max)
        if t_hi > 1:
            np.maximum(dp[:, f - 1, 1:t_hi], dp[:, f - 1, 0:t_hi - 1], out=dp[:, f, 1:t_hi])
            dp[:, f, 1:t_hi] += lp[:, f, 1:t_hi]

    durations = np.zeros((B, T_max), dtype=np.int64)
    attn_hard = np.zeros((B, F_max, T_max), dtype=np.float32)
    t_cur = np.maximum(txt_np - 1, 0)
    for f in range(F_max - 1, -1, -1):
        active = np.nonzero(f < mel_np)[0]
        if active.size == 0:
            continue
        tb = t_cur[active]
        durations[active, tb] += 1
        attn_hard[active, f, tb] = 1.0
        if f > 0:
            d_left = dp[active, f - 1, np.maximum(tb - 1, 0)]
            d_stay = dp[active, f - 1, tb]
            move = (tb > 0) & ((d_left > d_stay) | (f <= tb))
            t_cur[active] = tb - move
    return torch.from_numpy(attn_hard).to(attn_logprob.device), torch.from_numpy(durations).to(attn_logprob.device)


def expand_by_duration(h: torch.Tensor, durations: torch.Tensor, max_frames: int = None):
    B, _, D = h.shape
    outs, out_lens = [], []
    for b in range(B):
        rep = torch.repeat_interleave(h[b], durations[b].clamp(min=0), dim=0)
        if max_frames is not None:
            rep = F.pad(rep, (0, 0, 0, max_frames - rep.shape[0])) if rep.shape[0] < max_frames else rep[:max_frames]
        outs.append(rep); out_lens.append(rep.shape[0])
    F_max = max(out_lens) if max_frames is None else max_frames
    padded = torch.zeros(B, F_max, D, device=h.device, dtype=h.dtype)
    for b, rep in enumerate(outs):
        padded[b, :rep.shape[0]] = rep
    return padded, torch.tensor(out_lens, device=h.device, dtype=torch.long)


class CausalPredictorHead(nn.Module):
    def __init__(self, d_model: int = 256, kernel_size: int = 3):
        super().__init__()
        self.c1 = CausalConv1d(d_model, d_model, kernel_size)
        self.n1 = nn.LayerNorm(d_model)
        self.c2 = CausalConv1d(d_model, d_model, kernel_size)
        self.n2 = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.c1(x.transpose(1, 2)).transpose(1, 2)))
        h = F.silu(self.n2(self.c2(h.transpose(1, 2)).transpose(1, 2)))
        return self.proj(h).squeeze(-1)

    def init_state(self, batch_size: int, device, dtype):
        return (self.c1.init_state(batch_size, device, dtype), self.c2.init_state(batch_size, device, dtype))

    def forward_step(self, x: torch.Tensor, state):
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype), state
        h, s1 = self.c1.forward_step(x.transpose(1, 2), state[0])
        h = F.silu(self.n1(h.transpose(1, 2)))
        h, s2 = self.c2.forward_step(h.transpose(1, 2), state[1])
        h = F.silu(self.n2(h.transpose(1, 2)))
        return self.proj(h).squeeze(-1), (s1, s2)


class AetherAcousticModel(nn.Module):
    def __init__(self, vocab_size: int, backbone_type: str = "swa", d_model: int = 256, n_mels: int = 100, k_left: int = 3, lookahead: int = 4, swa_window: int = 48):
        super().__init__()
        self.backbone_type = backbone_type
        self.d_model = d_model
        self.n_mels = n_mels
        self.lookahead = lookahead
        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_emb = nn.Embedding(4, d_model)

        if backbone_type == "offline_transformer":
            self.backbone = OfflineBiTransformerBackbone(d_model=d_model, n_layers=4, n_heads=4, d_ff=896)
        elif backbone_type == "conv_gru":
            self.backbone = ConvUniGRUBackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_conv=4)
        elif backbone_type == "swa":
            self.backbone = SWABackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_layers=5, n_heads=4, window_size=swa_window)
        elif backbone_type == "ssm":
            self.backbone = SelectiveSSMBackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_layers=5)
        elif backbone_type == "rwkv":
            self.backbone = RWKVBackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_layers=5, n_heads=4)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        self.is_streaming = self.backbone.is_streaming
        self.aligner = AlignmentEncoder(n_mels=n_mels, d_text=d_model, d_attn=80)
        self.dur_predictor = CausalPredictorHead(d_model, kernel_size=3)
        nn.init.constant_(self.dur_predictor.proj.bias, math.log(9.0))
        self.f0_predictor = CausalPredictorHead(d_model, kernel_size=3)
        self.energy_predictor = CausalPredictorHead(d_model, kernel_size=3)
        self.f0_emb = nn.Linear(1, d_model)
        self.energy_emb = nn.Linear(1, d_model)
        self.frame_decoder = nn.ModuleList([CausalConvNeXtBlock(d_model, kernel_size=5, mult=2) for _ in range(4)])
        self.mel_out = nn.Linear(d_model, n_mels)

    def encode_text(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, text_mask: torch.Tensor = None):
        return self.backbone(self.phoneme_emb(phoneme_ids) + self.lang_emb(lang_ids), mask=text_mask)

    def forward_train(self, phoneme_ids, lang_ids, text_lens, mel_gt, f0_gt, energy_gt, mel_lens, step_ratio: float = 1.0):
        B, T_max = phoneme_ids.shape
        F_max = mel_gt.shape[-1]
        text_mask = torch.arange(T_max, device=phoneme_ids.device)[None, :] < text_lens[:, None]
        mel_mask = torch.arange(F_max, device=phoneme_ids.device)[None, :] < mel_lens[:, None]

        # Aligner 基于纯音素嵌入 + Beta-Binomial 对角先验，避免受流式右前瞻/因果偏移影响且严防 <bos> 坍缩
        x_emb = self.phoneme_emb(phoneme_ids) + self.lang_emb(lang_ids)
        h_txt = self.backbone(x_emb, mask=text_mask)
        attn_log_prior = build_beta_binomial_log_prior(text_lens, mel_lens, T_max, F_max, scaling=1.0)
        attn_logprob, attn_soft = self.aligner(mel_gt, x_emb, text_mask, attn_log_prior=attn_log_prior)
        loss_align = forward_sum_loss(attn_logprob, text_lens, mel_lens)
        attn_hard, dur_gt = viterbi_monotonic_alignment(attn_logprob, text_lens, mel_lens)

        # 前 25% 步数用 soft attention 让 mel 重建梯度直接训练对齐器；之后直通硬对齐 + 二值化损失，实现训练↔推理零落差
        use_hard = (not self.training) or (step_ratio >= 0.25)
        attn_use = (attn_hard + (attn_soft - attn_soft.detach())) if use_hard else attn_soft
        h_frame = torch.bmm(attn_use.to(h_txt.dtype), h_txt)

        loss_bin = -(attn_hard * attn_logprob * mel_mask[:, :, None]).sum() / mel_mask.sum().clamp(min=1)
        bin_weight = 0.0 if step_ratio < 0.25 else min(1.0, (step_ratio - 0.25) / 0.25)

        log_dur_pred = self.dur_predictor(h_txt)
        loss_dur = (F.l1_loss(log_dur_pred, torch.log(dur_gt.float() + 1.0), reduction="none") * text_mask).sum() / text_mask.sum().clamp(min=1)

        f0_pred = self.f0_predictor(h_frame)
        energy_pred = self.energy_predictor(h_frame)
        loss_f0 = (F.l1_loss(f0_pred, f0_gt, reduction="none") * mel_mask).sum() / mel_mask.sum().clamp(min=1)
        loss_energy = (F.l1_loss(energy_pred, energy_gt, reduction="none") * mel_mask).sum() / mel_mask.sum().clamp(min=1)

        # 验证期固定使用预测 F0/Energy（确定性衡量真实推理质量）；训练后期 50% 概率混合预测 F0/Energy 消除曝光偏差
        use_pred = (not self.training) or (step_ratio >= 0.25 and torch.rand(1).item() < 0.5)
        f0_in = f0_pred.detach() if use_pred else f0_gt
        e_in = energy_pred.detach() if use_pred else energy_gt
        h_dec = h_frame + self.f0_emb(f0_in.unsqueeze(-1)) + self.energy_emb(e_in.unsqueeze(-1))
        for blk in self.frame_decoder:
            h_dec = blk(h_dec)
        mel_pred = self.mel_out(h_dec).transpose(1, 2)

        loss_mel = (F.l1_loss(mel_pred, mel_gt, reduction="none") * mel_mask[:, None, :]).sum() / (mel_mask.sum().clamp(min=1) * self.n_mels)
        total_loss = loss_mel + 0.5 * loss_dur + 0.25 * loss_f0 + 0.25 * loss_energy + 0.5 * loss_align + 0.5 * bin_weight * loss_bin
        bos_ratio = (dur_gt[:, 0].float() / mel_lens.float().clamp(min=1)).mean().detach()
        return {
            "loss": total_loss, "mel": loss_mel.detach(), "dur": loss_dur.detach(),
            "f0": loss_f0.detach(), "energy": loss_energy.detach(),
            "align": loss_align.detach(), "bin": loss_bin.detach(), "bos_ratio": bos_ratio,
        }

    @torch.no_grad()
    def infer_offline(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, speed: float = 1.0):
        h_txt = self.encode_text(phoneme_ids, lang_ids)
        dur = torch.clamp(torch.round((torch.exp(self.dur_predictor(h_txt)) - 1.0) / speed), min=1, max=60).long()
        h_frame, _ = expand_by_duration(h_txt, dur)
        f0_pred = self.f0_predictor(h_frame)
        e_pred = self.energy_predictor(h_frame)
        h_dec = h_frame + self.f0_emb(f0_pred.unsqueeze(-1)) + self.energy_emb(e_pred.unsqueeze(-1))
        for blk in self.frame_decoder:
            h_dec = blk(h_dec)
        return self.mel_out(h_dec).transpose(1, 2), dur, f0_pred, e_pred

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        assert self.is_streaming
        device = device or next(self.parameters()).device
        return {
            "backbone": self.backbone.init_state(batch_size, device, dtype),
            "dur": self.dur_predictor.init_state(batch_size, device, dtype),
            "f0": self.f0_predictor.init_state(batch_size, device, dtype),
            "energy": self.energy_predictor.init_state(batch_size, device, dtype),
            "frame_dec": [blk.init_state(batch_size, device, dtype) for blk in self.frame_decoder],
        }

    @torch.no_grad()
    def infer_stream_step(self, phoneme_chunk: torch.Tensor, lang_chunk: torch.Tensor, state: dict, is_final: bool = False, speed: float = 1.0):
        x = (self.phoneme_emb(phoneme_chunk) + self.lang_emb(lang_chunk)) if phoneme_chunk.shape[1] > 0 else torch.zeros(phoneme_chunk.shape[0], 0, self.d_model, device=phoneme_chunk.device, dtype=self.phoneme_emb.weight.dtype)
        h_emit, state["backbone"] = self.backbone.forward_step(x, state["backbone"], is_final=is_final)
        if h_emit.shape[1] == 0:
            return torch.zeros(x.shape[0], self.n_mels, 0, device=x.device, dtype=x.dtype), state
        log_dur, state["dur"] = self.dur_predictor.forward_step(h_emit, state["dur"])
        dur = torch.clamp(torch.round((torch.exp(log_dur) - 1.0) / speed), min=1, max=60).long()
        h_frame, _ = expand_by_duration(h_emit, dur)
        f0_pred, state["f0"] = self.f0_predictor.forward_step(h_frame, state["f0"])
        e_pred, state["energy"] = self.energy_predictor.forward_step(h_frame, state["energy"])
        h_dec = h_frame + self.f0_emb(f0_pred.unsqueeze(-1)) + self.energy_emb(e_pred.unsqueeze(-1))
        new_fdec = []
        for blk, fst in zip(self.frame_decoder, state["frame_dec"]):
            h_dec, nfst = blk.forward_step(h_dec, fst)
            new_fdec.append(nfst)
        state["frame_dec"] = new_fdec
        return self.mel_out(h_dec).transpose(1, 2), state


# ==============================================================================
# 5.4 自研 24 kHz 因果流式声码器 CausalVocosLite (~3.6M) + 零爆音 OLA + 判别器
# ==============================================================================
class StreamingISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))

    def _frames_to_time(self, x: torch.Tensor) -> torch.Tensor:
        mag, p = self.out(x).float().chunk(2, dim=-1)
        mag = torch.exp(mag).clamp(max=1e2)
        spec = torch.complex(mag * torch.cos(p), mag * torch.sin(p))
        return torch.fft.irfft(spec, n=self.n_fft, dim=-1) * self.window[None, None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_frames, _ = x.shape
        frames = self._frames_to_time(x)
        pad = self.n_fft - self.hop_length
        total_len = n_frames * self.hop_length + pad
        out = torch.zeros(B, total_len, device=x.device, dtype=torch.float32)
        env = torch.zeros(B, total_len, device=x.device, dtype=torch.float32)
        w2 = (self.window ** 2)[None, :]
        for m in range(n_frames):
            st = m * self.hop_length
            out[:, st : st + self.n_fft] += frames[:, m, :]
            env[:, st : st + self.n_fft] += w2
        left_trim = pad // 2
        return out[:, left_trim : left_trim + n_frames * self.hop_length] / env[:, left_trim : left_trim + n_frames * self.hop_length].clamp(min=1e-6)

    def init_state(self, batch_size: int, device):
        pad = self.n_fft - self.hop_length
        return {
            "ola_out": torch.zeros(batch_size, pad, device=device, dtype=torch.float32),
            "ola_env": torch.zeros(batch_size, pad, device=device, dtype=torch.float32),
            "trim_remaining": pad // 2,
        }

    @torch.no_grad()
    def forward_step(self, x_chunk: torch.Tensor, state: dict, is_final: bool = False):
        B, C, _ = x_chunk.shape
        pad = self.n_fft - self.hop_length
        left_trim = pad // 2
        if C == 0 and not is_final:
            return torch.zeros(B, 0, device=x_chunk.device, dtype=torch.float32), state
        out_len = C * self.hop_length + pad
        out = torch.zeros(B, out_len, device=x_chunk.device, dtype=torch.float32)
        env = torch.zeros(B, out_len, device=x_chunk.device, dtype=torch.float32)
        out[:, :pad] = state["ola_out"]
        env[:, :pad] = state["ola_env"]
        if C > 0:
            frames = self._frames_to_time(x_chunk)
            w2 = (self.window ** 2)[None, :]
            for m in range(C):
                st = m * self.hop_length
                out[:, st : st + self.n_fft] += frames[:, m, :]
                env[:, st : st + self.n_fft] += w2
        emit_end = C * self.hop_length
        trim_rem = state["trim_remaining"]
        if is_final:
            tail_end = emit_end + (pad - left_trim)
            wav_emit = torch.zeros(B, 0, device=x_chunk.device, dtype=torch.float32) if tail_end <= trim_rem else out[:, trim_rem:tail_end] / env[:, trim_rem:tail_end].clamp(min=1e-6)
            return wav_emit, self.init_state(B, x_chunk.device)
        if emit_end <= trim_rem:
            wav_emit = torch.zeros(B, 0, device=x_chunk.device, dtype=torch.float32)
            new_trim = trim_rem - emit_end
        else:
            wav_emit = out[:, trim_rem:emit_end] / env[:, trim_rem:emit_end].clamp(min=1e-6)
            new_trim = 0
        return wav_emit, {"ola_out": out[:, emit_end : emit_end + pad].clone(), "ola_env": env[:, emit_end : emit_end + pad].clone(), "trim_remaining": new_trim}


class CausalVocosLite(nn.Module):
    def __init__(self, n_mels: int = 100, dim: int = 256, n_layers: int = 8, kernel_size: int = 7, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.embed = CausalConv1d(n_mels, dim, kernel_size=7)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([CausalConvNeXtBlock(dim, kernel_size=kernel_size, mult=3) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = StreamingISTFTHead(dim, n_fft=n_fft, hop_length=hop_length)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(self.embed(mel).transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_out(h))

    def init_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {"embed": self.embed.init_state(batch_size, device, dtype), "blocks": [b.init_state(batch_size, device, dtype) for b in self.blocks], "head": self.head.init_state(batch_size, device)}

    @torch.no_grad()
    def forward_step(self, mel_chunk: torch.Tensor, state: dict, is_final: bool = False):
        if mel_chunk.shape[-1] == 0:
            h = torch.zeros(mel_chunk.shape[0], 0, self.norm_out.normalized_shape[0], device=mel_chunk.device, dtype=mel_chunk.dtype)
            wav, state["head"] = self.head.forward_step(h, state["head"], is_final=is_final)
            return wav, state
        h, state["embed"] = self.embed.forward_step(mel_chunk, state["embed"])
        h = self.norm_in(h.transpose(1, 2))
        new_blks = []
        for blk, bst in zip(self.blocks, state["blocks"]):
            h, nb = blk.forward_step(h, bst)
            new_blks.append(nb)
        state["blocks"] = new_blks
        wav, state["head"] = self.head.forward_step(self.norm_out(h), state["head"], is_final=is_final)
        return wav, state


class PeriodDisc(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0)), nn.Conv2d(32, 64, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0)), nn.Conv2d(128, 256, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(256, 1, (3, 1), 1, (1, 0)),
        ])

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - T % self.period), mode="reflect")
        x = x.view(B, 1, -1, self.period)
        fmap = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        return x.flatten(1, -1), fmap


class SpecDisc(nn.Module):
    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft))
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4)), nn.Conv2d(32, 64, (3, 9), (1, 2), (1, 4)),
            nn.Conv2d(64, 128, (3, 9), (1, 2), (1, 4)), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.Conv2d(128, 1, (3, 3), (1, 1), (1, 1)),
        ])

    def forward(self, x: torch.Tensor):
        h = torch.stft(x.float(), self.n_fft, self.hop, self.n_fft, self.win, return_complex=True).abs().unsqueeze(1)
        fmap = []
        for i, conv in enumerate(self.convs):
            h = conv(h)
            if i < len(self.convs) - 1:
                h = F.leaky_relu(h, 0.1)
            fmap.append(h)
        return h.flatten(1, -1), fmap


class LiteVocoderDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([
            PeriodDisc(2), PeriodDisc(3), PeriodDisc(5), PeriodDisc(7),
            SpecDisc(512, 128), SpecDisc(1024, 256), SpecDisc(2048, 512),
        ])

    def forward(self, y_real: torch.Tensor, y_fake: torch.Tensor):
        ro_l, fo_l, rf_l, ff_l = [], [], [], []
        for d in self.discs:
            ro, rf = d(y_real); fo, ff = d(y_fake)
            ro_l.append(ro); fo_l.append(fo); rf_l.append(rf); ff_l.append(ff)
        return ro_l, fo_l, rf_l, ff_l


def multi_res_stft_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        win = torch.hann_window(n_fft, device=y_pred.device)
        sp = torch.stft(y_pred.float(), n_fft, hop, n_fft, win, return_complex=True).abs().clamp(min=1e-5)
        st = torch.stft(y_true.float(), n_fft, hop, n_fft, win, return_complex=True).abs().clamp(min=1e-5)
        loss = loss + torch.norm(st - sp, p="fro") / (torch.norm(st, p="fro") + 1e-6) + F.l1_loss(torch.log(sp), torch.log(st))
    return loss / 3.0


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ==============================================================================
# 6b. 向量化热路径 vs 参考实现 逐位等价自检（--selftest）
# ==============================================================================
def run_aligner_equivalence_selftest():
    print("=" * 72)
    print("向量化对齐热路径 vs notebook 参考实现 —— 逐位等价自检 (torch.equal)")
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
            # 说明：lgamma 在 CPU/CUDA 间允许 <=1 ulp 差异（notebook 训练时前后一致地在
            # 同一设备上计算先验，故只需同设备逐位一致；此处仅做宽松一致性确认）
            assert torch.allclose(p_ref_d.cpu(), p_ref, rtol=1e-5, atol=1e-4), f"prior cross-device drift seed={seed}"
        print(f"  [PASS] seed={seed}  shape=({B},{F_max},{T_max}) CPU+CUDA 逐位一致")
    print("[SelfTest] 全部通过：向量化实现与 notebook 参考实现完全等价。")
    print("=" * 72)


if SELFTEST:
    run_aligner_equivalence_selftest()


# #############################################################################
# Cell 6: 训练前门禁
# #############################################################################
@torch.no_grad()
def run_architecture_and_streaming_tests():
    print("=" * 72)
    print("参数量检查 & 流式↔离线等价性测试 (要求最大绝对误差 < 1e-4)")
    print("=" * 72)
    T = 23
    pids = torch.randint(4, FRONTEND.vocab_size, (1, T), dtype=torch.long)
    lids = torch.randint(0, 3, (1, T), dtype=torch.long)

    for exp_name, bb_type, lookahead in EXPERIMENTS:
        m = AetherAcousticModel(
            vocab_size=FRONTEND.vocab_size, backbone_type=bb_type, d_model=CFG["d_model"],
            n_mels=CFG["n_mels"], k_left=CFG["k_left"], lookahead=lookahead, swa_window=CFG["swa_window"],
        ).eval()
        infer_params = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("aligner.")) / 1e6
        mel_off, _, _, _ = m.infer_offline(pids, lids)
        if m.is_streaming:
            st = m.init_stream_state(1, device=pids.device, dtype=torch.float32)
            chunks = []
            step = CFG["stream_text_chunk"]
            for i in range(0, T, step):
                mc, st = m.infer_stream_step(pids[:, i:i+step], lids[:, i:i+step], st, is_final=(i + step >= T))
                if mc.shape[-1] > 0:
                    chunks.append(mc)
            mel_str = torch.cat(chunks, dim=-1)
            diff = (mel_off - mel_str).abs().max().item()
            assert mel_str.shape == mel_off.shape and diff < 1e-4, f"{exp_name} streaming mismatch: {diff}"
            print(f"  [PASS] {exp_name:26s} | Infer: {infer_params:.2f}M | L={lookahead} | Stream Diff: {diff:.2e}")
        else:
            print(f"  [PASS] {exp_name:26s} | Infer: {infer_params:.2f}M | (Offline Upper-Bound)")

    voc = CausalVocosLite(n_mels=CFG["n_mels"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"]).eval()
    mel_dummy = torch.randn(1, CFG["n_mels"], 53)
    wav_off = voc(mel_dummy)
    vst = voc.init_state(1, device=mel_dummy.device, dtype=torch.float32)
    w_chunks = []
    for i in range(0, 53, CFG["stream_mel_chunk"]):
        wc, vst = voc.forward_step(mel_dummy[:, :, i:i+CFG["stream_mel_chunk"]], vst, is_final=(i + CFG["stream_mel_chunk"] >= 53))
        if wc.shape[-1] > 0:
            w_chunks.append(wc)
    vdiff = (wav_off - torch.cat(w_chunks, dim=-1)).abs().max().item()
    assert vdiff < 1e-4, f"Vocoder OLA mismatch: {vdiff}"
    print(f"  [PASS] {'CausalVocosLite (24kHz)':26s} | Infer: {count_params_m(voc):.2f}M | OLA Diff: {vdiff:.2e}")
    print("=" * 72)


run_architecture_and_streaming_tests()

if SELFTEST:
    print('[SelfTest] 架构与流式等价门禁已通过，退出 --selftest 模式。')
    import sys as _sys
    _sys.exit(0)

# #############################################################################
# Cell 4: 数据下载与预处理
# #############################################################################
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download


def safe_extract_tar(archive_path: Path, dest_dir: Path, desc: str = "Extract"):
    """安全高速解压 tar / tar.bz2 / tar.gz：优先走本地 GNU tar（带断点进度、低内存），回退流式 tarfile(..., filter='data') 彻底消除 Python 3.14 DeprecationWarning"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name_l = archive_path.name.lower()
    is_bz2 = name_l.endswith((".tar.bz2", ".tbz2", ".bz2"))
    is_gz = name_l.endswith((".tar.gz", ".tgz", ".gz"))

    use_gnu_tar = False
    if sys.platform != "win32" and shutil.which("tar"):
        try:
            ver_out = subprocess.check_output(["tar", "--version"], text=True, stderr=subprocess.STDOUT)
            use_gnu_tar = ("GNU tar" in ver_out)
        except Exception:
            use_gnu_tar = False

    if use_gnu_tar:
        cmd = ["tar", "--no-same-owner"]
        if is_bz2:
            bzip_prog = shutil.which("lbzip2") or shutil.which("pbzip2")
            if bzip_prog:
                cmd.append(f"--use-compress-program={bzip_prog}")
            else:
                cmd.append("-j")
        elif is_gz:
            cmd.append("-z")
        cmd.extend([
            "--checkpoint=2500",
            "--checkpoint-action=echo=tar_checkpoint:%u",
            "-xf", str(archive_path),
            "-C", str(dest_dir),
        ])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_err_lines = []
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if "tar_checkpoint:" in line:
                cnt = line.rsplit(":", 1)[-1]
                print(f"  [{desc}] 已解压 {cnt} 个归档块...", flush=True)
            elif line:
                last_err_lines.append(line)
                if len(last_err_lines) > 10:
                    last_err_lines.pop(0)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"tar 解压失败 (exit={rc}): {' | '.join(last_err_lines)}")
        return

    # 便携回退：使用流式模式 ("r|...") + filter="data"，O(1) 内存且无 Python 3.14 DeprecationWarning
    rmode = "r|bz2" if is_bz2 else ("r|gz" if is_gz else ("r|" if name_l.endswith(".tar") else "r|*"))
    extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(archive_path, rmode) as tf:
        tf.extractall(dest_dir, **extract_kwargs)


def _extract_possibly_drive_archive(storage_archive: Path, local_tmp_dir: Path, dest_dir: Path, desc: str):
    """若归档位于云盘挂载路径（Colab DriveFS），先按大块复制到本地 SSD 再解压，严禁直接在 DriveFS 上解压 GB 级包"""
    if IS_LOCAL or not _is_subpath(storage_archive, STORAGE_ROOT):
        safe_extract_tar(storage_archive, dest_dir, desc=desc)
        return
    staged = local_tmp_dir / storage_archive.name
    try:
        print(f"[Stage] 将云盘归档分块复制到本地 SSD 再解压: {storage_archive.name}", flush=True)
        copy_from_storage_to_local(storage_archive, staged)
        safe_extract_tar(staged, dest_dir, desc=desc)
    finally:
        staged.unlink(missing_ok=True)


def _extract_single_item(args):
    it, pids, lids, feat_file_str, cfg_sub = args
    feat_file = Path(feat_file_str)
    if feat_file.exists():
        try:
            d = torch.load(feat_file, map_location="cpu", weights_only=False)
            wav_t = decode_wav_tensor(d["wav"])
            dur_sec = float(wav_t.shape[0]) / cfg_sub["sample_rate"]
            return (
                it, feat_file_str, dur_sec,
                d["mel"].float().numpy(),
                d["log_f0"].float().numpy(),
                d["log_energy"].float().numpy(),
                d["uv"].float().numpy(),
                len(d["phoneme_ids"]), int(d["mel"].shape[-1]),
            )
        except Exception:
            feat_file.unlink(missing_ok=True)
    try:
        wav = load_and_normalize_wav(Path(it["wav_path"]), cfg_sub["sample_rate"])
    except Exception:
        return None
    dur_sec = len(wav) / cfg_sub["sample_rate"]
    if dur_sec < cfg_sub["min_audio_sec"] or dur_sec > cfg_sub["max_audio_sec"]:
        return None
    if len(pids) < 4 or len(pids) > 220:
        return None
    wav_t = torch.from_numpy(wav)
    with torch.no_grad():
        mel_t = MEL_EXTRACTOR(wav_t)
    n_frames = mel_t.shape[-1]
    if n_frames < len(pids) * 2:
        return None
    f0_np, e_np, uv_np = extract_f0_and_energy(wav, cfg_sub["sample_rate"], cfg_sub["hop_length"], n_frames)
    wav_i16 = torch.clamp(torch.round(wav_t * 32767.0), -32768, 32767).to(torch.int16)
    d = {
        "utt_id": it["utt_id"],
        "text": it["text"],
        "phoneme_ids": torch.tensor(pids, dtype=torch.long),
        "lang_ids": torch.tensor(lids, dtype=torch.long),
        "mel": mel_t.to(torch.float16),
        "log_f0": torch.from_numpy(f0_np),
        "log_energy": torch.from_numpy(e_np),
        "uv": torch.from_numpy(uv_np),
        "wav": wav_i16,
    }
    atomic_torch_save(d, feat_file)
    return (it, feat_file_str, dur_sec, mel_t.to(torch.float32).numpy(), f0_np, e_np, uv_np, len(pids), int(n_frames))


def download_and_prepare_dataset(cfg):
    src = cfg["dataset_source"]
    tag = f"{src}_v2_max{cfg['max_train_hours']}h"
    storage_raw_dir = STORAGE_ROOT / "datasets" / "raw"
    local_raw_dir = LOCAL_WORK / "data" / src
    local_feat_dir = LOCAL_WORK / "cache" / tag
    local_manifest = local_feat_dir / "manifest.json"
    storage_feat_tar = STORAGE_ROOT / "datasets" / f"{tag}_features.tar"
    storage_shards_dir = STORAGE_ROOT / "datasets" / f"{tag}_shards"
    storage_shards_meta = storage_shards_dir / "shards_meta.json"

    local_raw_dir.mkdir(parents=True, exist_ok=True)
    local_feat_dir.mkdir(parents=True, exist_ok=True)

    # 快速通道 1：本地已预处理完，秒加载！
    if local_manifest.exists():
        meta = json.loads(local_manifest.read_text(encoding="utf-8"))
        if meta.get("train") and Path(meta["train"][0]["feat_path"]).exists():
            print(f"[Cache Hit] 直接加载本地预处理特征集: {local_manifest} ({meta['total_hours']}h, train={meta['train_count']}, val={meta['val_count']})")
            save_progress_status("dataset", {"status": "cached_local", "hours": meta["total_hours"], "train": meta["train_count"], "val": meta["val_count"]})
            return meta

    # 快速通道 2（Colab 断连重连）：优先从云盘分片缓存 (~250MB/片) 或兼容旧版单文件缓存快速恢复
    if SYNC_FEATURES_TAR_TO_DRIVE and (storage_shards_meta.exists() or storage_feat_tar.exists()):
        restored_ok = False
        if storage_shards_meta.exists():
            print(f"[Drive Feature Cache Hit] 发现云盘分片特征缓存，正在快速解压恢复: {storage_shards_dir}", flush=True)
            try:
                smeta = json.loads(storage_shards_meta.read_text(encoding="utf-8"))
                for s_info in tqdm(smeta.get("shards", []), desc="Restore Drive Shards", mininterval=1.0):
                    shard_src = storage_shards_dir / s_info["name"]
                    assert shard_src.exists(), f"Missing shard: {shard_src}"
                    _extract_possibly_drive_archive(shard_src, LOCAL_WORK / "tmp_dl", local_feat_dir.parent, desc=s_info["name"])
                restored_ok = True
            except Exception as e:
                print(f"[Warn] 云盘分片缓存恢复失败 ({e})，将改用原始数据集重新预处理。", flush=True)
        elif storage_feat_tar.exists():
            print(f"[Drive Feature Cache Hit] 发现云盘特征缓存，正在快速解压恢复: {storage_feat_tar}", flush=True)
            try:
                _extract_possibly_drive_archive(storage_feat_tar, LOCAL_WORK / "tmp_dl", local_feat_dir.parent, desc=storage_feat_tar.name)
                restored_ok = True
            except Exception as e:
                print(f"[Warn] 云盘单文件缓存恢复失败 ({e})，将改用原始数据集重新预处理。", flush=True)

        if restored_ok and local_manifest.exists():
            meta = json.loads(local_manifest.read_text(encoding="utf-8"))
            for split in ("train", "val"):
                for e in meta[split]:
                    e["feat_path"] = str(local_feat_dir / Path(e["feat_path"]).name)
            if meta.get("train") and Path(meta["train"][0]["feat_path"]).exists():
                atomic_write_text(json.dumps(meta, ensure_ascii=False, indent=2), local_manifest)
                print(f"[Restored] 秒恢复 {meta['total_hours']}h 预处理特征 (train={meta['train_count']}, val={meta['val_count']})")
                save_progress_status("dataset", {"status": "restored_from_drive", "hours": meta["total_hours"], "train": meta["train_count"], "val": meta["val_count"]})
                return meta

    items = []
    if src == "bznsyp":
        archive_path = storage_raw_dir / "BZNSYP.tar.bz2"
        marker = local_raw_dir / ".extracted"
        if marker.exists():
            has_txt = bool(list(local_raw_dir.rglob("*000001-010000.txt")))
            has_wav = any(p.is_dir() for p in local_raw_dir.rglob("Wave"))
            if not (has_txt and has_wav):
                marker.unlink(missing_ok=True)

        if not marker.exists():
            dl_dir = storage_raw_dir if IS_LOCAL else (LOCAL_WORK / "tmp_dl")
            dl_dir.mkdir(parents=True, exist_ok=True)
            extracted_ok = False

            if archive_path.exists() and archive_path.stat().st_size > 100_000_000:
                print(f"[Cache Hit] 复用原始压缩包: {archive_path}", flush=True)
                try:
                    _extract_possibly_drive_archive(archive_path, dl_dir, local_raw_dir, desc="BZNSYP")
                    extracted_ok = True
                except Exception as e:
                    print(f"[Warn] 已有压缩包解压失败 ({e})，将直接从 HuggingFace 下载到本地高速盘...", flush=True)
                    shutil.rmtree(local_raw_dir, ignore_errors=True)
                    local_raw_dir.mkdir(parents=True, exist_ok=True)

            if not extracted_ok:
                print(f"[Download] 从 HuggingFace 下载标贝 BZNSYP (12h) 到本地目录 {dl_dir} ...", flush=True)
                dl_path = Path(hf_hub_download(
                    repo_id="openspeech/BZNSYP",
                    filename="BZNSYP.tar.bz2",
                    repo_type="model",
                    local_dir=str(dl_dir),
                ))
                print(f"[Extract] 解压 BZNSYP 到 {local_raw_dir} ...", flush=True)
                safe_extract_tar(dl_path, local_raw_dir, desc="BZNSYP")
                if CACHE_RAW_TO_DRIVE and not _same_path(dl_path, archive_path):
                    print(f"[Cache] 同步原始压缩包到 {archive_path} ...", flush=True)
                    sync_file_to_storage(dl_path, archive_path)
                if not IS_LOCAL:
                    shutil.rmtree(dl_dir, ignore_errors=True)
            marker.touch()

        txt_candidates = list(local_raw_dir.rglob("*000001-010000.txt"))
        assert txt_candidates, f"未找到 000001-010000.txt，请检查目录 {local_raw_dir}"
        label_file = txt_candidates[0]
        wav_dirs = [p for p in local_raw_dir.rglob("Wave") if p.is_dir()]
        assert wav_dirs, f"未找到 Wave 目录，请检查目录 {local_raw_dir}"
        wav_dir = wav_dirs[0]

        raw_bytes = label_file.read_bytes()
        try:
            lines = raw_bytes.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            lines = raw_bytes.decode("gb18030", errors="ignore").splitlines()

        def _merge_bznsyp_punc(clean_txt: str, pys: list):
            norm_t = FRONTEND.normalize_text(clean_txt)
            hanzi_cnt = len(re.findall(r"[\u4e00-\u9fff]", norm_t))
            if hanzi_cnt == len(pys):
                merged, pi = [], 0
                for ch in norm_t:
                    if re.match(r"[\u4e00-\u9fff]", ch):
                        merged.append(pys[pi]); pi += 1
                    elif ch in (",", ".", "?", "!", ";", ":"):
                        merged.append(ch)
                return merged
            # 含儿化音字数不等时保留原始拼音并补齐句末标点
            out_py = list(pys)
            if norm_t and norm_t[-1] in (".", "?", "!"):
                out_py.append(norm_t[-1])
            return out_py

        for i in range(0, len(lines) - 1, 2):
            line_txt, line_py = lines[i].strip(), lines[i + 1].strip()
            if not line_txt or "\t" not in line_txt:
                continue
            utt_id, text_with_prosody = line_txt.split("\t", 1)
            wav_path = wav_dir / f"{utt_id.strip()}.wav"
            if wav_path.exists():
                clean_text = re.sub(r"#[1-4]", "", text_with_prosody).strip()
                items.append({
                    "utt_id": utt_id.strip(),
                    "wav_path": str(wav_path),
                    "text": clean_text,
                    "prosody_pinyin": _merge_bznsyp_punc(clean_text, line_py.split()),
                })

    elif src == "ljspeech":
        archive_path = storage_raw_dir / "LJSpeech-1.1.tar.bz2"
        marker = local_raw_dir / ".extracted"
        if marker.exists() and not (local_raw_dir / "LJSpeech-1.1" / "metadata.csv").exists():
            marker.unlink(missing_ok=True)

        if not marker.exists():
            dl_dir = storage_raw_dir if IS_LOCAL else (LOCAL_WORK / "tmp_dl")
            dl_dir.mkdir(parents=True, exist_ok=True)
            extracted_ok = False

            if archive_path.exists() and archive_path.stat().st_size > 100_000_000:
                print(f"[Cache Hit] 复用原始压缩包: {archive_path}", flush=True)
                try:
                    _extract_possibly_drive_archive(archive_path, dl_dir, local_raw_dir, desc="LJSpeech")
                    extracted_ok = True
                except Exception as e:
                    print(f"[Warn] 已有压缩包解压失败 ({e})，将重新下载到本地高速盘...", flush=True)
                    shutil.rmtree(local_raw_dir, ignore_errors=True)
                    local_raw_dir.mkdir(parents=True, exist_ok=True)

            if not extracted_ok:
                print(f"[Download] 下载 LJSpeech-1.1 (Public Domain) 到 {dl_dir} ...", flush=True)
                dl_path = dl_dir / "LJSpeech-1.1.tar.bz2"
                tmp_part = dl_path.with_suffix(".bz2.part")
                url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
                if shutil.which("wget"):
                    ret = subprocess.call(["wget", "-q", "--show-progress", "-O", str(tmp_part), url])
                    assert ret == 0 and tmp_part.exists(), "LJSpeech 下载失败，请检查网络连接"
                else:
                    import urllib.request
                    urllib.request.urlretrieve(url, str(tmp_part))
                os.replace(tmp_part, dl_path)
                print(f"[Extract] 解压 LJSpeech-1.1 到 {local_raw_dir} ...", flush=True)
                safe_extract_tar(dl_path, local_raw_dir, desc="LJSpeech")
                if CACHE_RAW_TO_DRIVE and not _same_path(dl_path, archive_path):
                    sync_file_to_storage(dl_path, archive_path)
                if not IS_LOCAL:
                    shutil.rmtree(dl_dir, ignore_errors=True)
            marker.touch()

        lj_root = local_raw_dir / "LJSpeech-1.1"
        for line in (lj_root / "metadata.csv").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[2] if len(parts) >= 3 else parts[1]
            wav_path = lj_root / "wavs" / f"{utt_id}.wav"
            if wav_path.exists():
                items.append({"utt_id": utt_id, "wav_path": str(wav_path), "text": text, "prosody_pinyin": None})

    elif src == "custom_bilingual":
        custom_dir = STORAGE_ROOT / "datasets" / "custom_bilingual"
        meta_path = custom_dir / "metadata.csv"
        assert meta_path.exists(), f"请先将单音色双语数据放到 {custom_dir} (包含 metadata.csv 与 wavs/)"
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            utt_id, text = parts[0].strip(), parts[1].strip()
            wav_path = custom_dir / "wavs" / f"{utt_id}.wav"
            if wav_path.exists():
                items.append({"utt_id": utt_id, "wav_path": str(wav_path), "text": text, "prosody_pinyin": None})
    else:
        raise ValueError(f"Unsupported DATASET_SOURCE: {src}")

    rng = random.Random(cfg["seed"])
    rng.shuffle(items)
    # 估算需要抽多少条以覆盖 max_train_hours + 0.5h，减少多余特征计算
    max_total_sec = (cfg["max_train_hours"] + 0.5) * 3600.0
    est_needed = min(len(items), int(max_total_sec / 2.6) + 400)
    cand_items = items[:est_needed]

    cfg_sub = {k: cfg[k] for k in ("sample_rate", "hop_length", "min_audio_sec", "max_audio_sec")}
    # 先在主线程完成极速文本 G2P，确保 pypinyin / g2p_en 100% 线程安全且无多进程 pickle/fork 风险
    tasks = []
    for it in cand_items:
        feat_file_str = str(local_feat_dir / f"{it['utt_id']}.pt")
        if Path(feat_file_str).exists():
            tasks.append((it, [], [], feat_file_str, cfg_sub))
        else:
            pids, lids = FRONTEND.text_to_ids(it["text"], prosody_pinyin=it["prosody_pinyin"])
            tasks.append((it, pids, lids, feat_file_str, cfg_sub))

    print(f"[Preprocess] 开始提取 24 kHz Mel+F0+Energy（候选 {len(tasks)} 条，目标上限 {cfg['max_train_hours']}h）...", flush=True)
    processed = []
    total_sec = 0.0
    mel_sum = np.zeros(cfg["n_mels"], dtype=np.float64)
    mel_sq = np.zeros(cfg["n_mels"], dtype=np.float64)
    mel_cnt = 0
    f0_vals, e_vals = [], []

    def _consume(res):
        nonlocal total_sec, mel_cnt
        if res is None:
            return
        it, fpath, dur_sec, mel_np, f0_np, e_np, uv_np, n_ph, n_fr = res
        if total_sec >= max_total_sec:
            return
        total_sec += dur_sec
        mel_f64 = np.asarray(mel_np, dtype=np.float64)
        mel_sum[:] += mel_f64.sum(axis=-1)
        mel_sq[:] += (mel_f64 ** 2).sum(axis=-1)
        mel_cnt += mel_f64.shape[-1]
        if uv_np.sum() > 0:
            f0_vals.append(f0_np[uv_np > 0.5])
        e_vals.append(e_np)
        processed.append({
            "utt_id": it["utt_id"],
            "text": it["text"],
            "feat_path": fpath,
            "dur_sec": dur_sec,
            "n_phonemes": int(n_ph),
            "n_frames": int(n_fr),
        })

    # 在主线程预热一次 DSP/Torch STFT，确保所有懒加载与 JIT 编译完成后再开多线程池
    _dummy_wav = np.zeros(int(0.2 * cfg["sample_rate"]), dtype=np.float32)
    with torch.no_grad():
        _ = MEL_EXTRACTOR(torch.from_numpy(_dummy_wav))
    _ = extract_f0_and_energy(_dummy_wav, cfg["sample_rate"], cfg["hop_length"], 10)
    del _dummy_wav

    prev_threads = torch.get_num_threads()
    prep_workers = min(4 if IS_COLAB else 8, max(2, CPU_CORES // 2 if CPU_CORES >= 4 else CPU_CORES))
    try:
        if prep_workers >= 2 and len(tasks) > 0:
            torch.set_num_threads(1)
            print(f"[Preprocess] 启用 {prep_workers} 线程并行提取 24 kHz 音频特征（单线程 Torch 防内存膨胀）...", flush=True)
            pbar = tqdm(total=len(tasks), desc="Parallel Extract", mininterval=2.0)
            done_cnt = 0
            with ThreadPoolExecutor(max_workers=prep_workers) as ex:
                batch_sz = 200
                for st in range(0, len(tasks), batch_sz):
                    if total_sec >= max_total_sec:
                        break
                    sub = tasks[st : st + batch_sz]
                    for res in ex.map(_extract_single_item, sub):
                        _consume(res)
                        pbar.update(1)
                        done_cnt += 1
                    if done_cnt % 600 == 0:
                        pbar.set_postfix(hours=f"{total_sec/3600:.2f}/{cfg['max_train_hours']+0.5:.1f}")
                        trim_process_memory()
                    if done_cnt % 1200 == 0:
                        print(f"  [Preprocess] 已处理 {done_cnt}/{len(tasks)} 条，有效累计 {total_sec/3600:.2f}h", flush=True)
            pbar.close()
        else:
            pbar = tqdm(tasks, desc="Extract 24kHz Mel+F0+Energy", mininterval=2.0)
            for idx_t, t in enumerate(pbar, 1):
                if total_sec >= max_total_sec:
                    break
                _consume(_extract_single_item(t))
                if idx_t % 500 == 0:
                    pbar.set_postfix(hours=f"{total_sec/3600:.2f}/{cfg['max_train_hours']+0.5:.1f}")
                    trim_process_memory()
    finally:
        torch.set_num_threads(prev_threads)

    del tasks, cand_items, items
    trim_process_memory()

    mel_mean = (mel_sum / max(mel_cnt, 1)).astype(np.float32)
    mel_std = np.sqrt(np.maximum(mel_sq / max(mel_cnt, 1) - mel_mean ** 2, 1e-4)).astype(np.float32)
    all_f0 = np.concatenate(f0_vals) if f0_vals else np.array([5.0], dtype=np.float32)
    all_e = np.concatenate(e_vals) if e_vals else np.array([-3.0], dtype=np.float32)

    stats = {
        "mel_mean": mel_mean.tolist(),
        "mel_std": mel_std.tolist(),
        "f0_mean": float(np.mean(all_f0)),
        "f0_std": float(max(np.std(all_f0), 1e-3)),
        "e_mean": float(np.mean(all_e)),
        "e_std": float(max(np.std(all_e), 1e-3)),
    }

    n_val = min(cfg["val_utts"], max(10, len(processed) // 10))
    val_list = processed[:n_val]
    train_list = processed[n_val:]

    meta = {
        "source": src,
        "total_hours": round(total_sec / 3600.0, 3),
        "train_count": len(train_list),
        "val_count": len(val_list),
        "stats": stats,
        "train": train_list,
        "val": val_list,
    }
    atomic_write_text(json.dumps(meta, ensure_ascii=False, indent=2), local_manifest)
    if not IS_LOCAL:
        sync_file_to_storage(local_manifest, STORAGE_ROOT / "datasets" / f"{tag}_manifest.json")
        # 特征已提取完毕，Colab/Kaggle 下清理临时解压的 48kHz 原始 WAV 释放约 4.5GB 本地空间
        if not _is_subpath(local_raw_dir, STORAGE_ROOT):
            shutil.rmtree(local_raw_dir, ignore_errors=True)

    # Colab 下把特征目录按 ~250MB 分片打包备份到 Google Drive（避免单写 4GB+ 大文件撑爆 DriveFS 内存/磁盘）
    if SYNC_FEATURES_TAR_TO_DRIVE and not storage_shards_meta.exists():
        print(f"[Backup] 将预处理特征分片 (~250MB/片) 备份至 Google Drive ({storage_shards_dir})...", flush=True)
        try:
            storage_shards_dir.mkdir(parents=True, exist_ok=True)
            all_files = sorted([p for p in local_feat_dir.iterdir() if p.is_file()])
            max_shard_bytes = 250 * 1024 * 1024
            shards, cur_group, cur_bytes = [], [], 0
            for fp in all_files:
                fsz = fp.stat().st_size
                if cur_group and cur_bytes + fsz > max_shard_bytes:
                    shards.append(cur_group)
                    cur_group, cur_bytes = [], 0
                cur_group.append(fp)
                cur_bytes += fsz
            if cur_group:
                shards.append(cur_group)

            shard_records = []
            for idx, grp in enumerate(tqdm(shards, desc="Backup Shards to Drive", mininterval=1.0)):
                sname = f"part_{idx:03d}.tar"
                dst_shard = storage_shards_dir / sname
                if dst_shard.exists() and dst_shard.stat().st_size > 0:
                    shard_records.append({"name": sname, "files": len(grp), "bytes": dst_shard.stat().st_size})
                    continue
                tmp_shard = LOCAL_WORK / f"{tag}_{sname}"
                with tarfile.open(tmp_shard, "w") as tf:
                    for fp in grp:
                        tf.add(fp, arcname=f"{local_feat_dir.name}/{fp.name}")
                sync_file_to_storage(tmp_shard, dst_shard)
                shard_records.append({"name": sname, "files": len(grp), "bytes": tmp_shard.stat().st_size})
                tmp_shard.unlink(missing_ok=True)
                trim_process_memory()

            meta_local = LOCAL_WORK / f"{tag}_shards_meta.json"
            atomic_write_text(json.dumps({"shards": shard_records, "total_files": len(all_files)}, indent=2), meta_local)
            sync_file_to_storage(meta_local, storage_shards_meta)
            meta_local.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Backup Warn] 云盘特征分片备份未完成 ({e})，不影响当前本地高速盘继续训练。", flush=True)

    save_progress_status("dataset", {"status": "ready", "hours": meta["total_hours"], "train": len(train_list), "val": len(val_list)})
    print(f"[Ready] 有效数据 {len(processed)} 句（共 {total_sec/3600:.2f} 小时）| Train={len(train_list)} | Val={len(val_list)}")
    return meta


DATASET_META = download_and_prepare_dataset(CFG)

# #############################################################################
# Cell 7: DataLoader
# #############################################################################
from torch.utils.data import Dataset, DataLoader

STATS = DATASET_META["stats"]
MEL_MEAN = torch.tensor(STATS["mel_mean"], dtype=torch.float32)[:, None]
MEL_STD = torch.tensor(STATS["mel_std"], dtype=torch.float32)[:, None]


def norm_mel(m): return (m - MEL_MEAN.to(m.device)) / MEL_STD.to(m.device)
def denorm_mel(m): return m * MEL_STD.to(m.device) + MEL_MEAN.to(m.device)


class AcousticDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        d = torch.load(self.entries[idx]["feat_path"], map_location="cpu", weights_only=False)
        return {
            "phoneme_ids": d["phoneme_ids"],
            "lang_ids": d["lang_ids"],
            "mel": (d["mel"].float() - MEL_MEAN) / MEL_STD,
            "f0": (d["log_f0"].float() - STATS["f0_mean"]) / STATS["f0_std"],
            "energy": (d["log_energy"].float() - STATS["e_mean"]) / STATS["e_std"],
        }


def acoustic_collate(batch):
    B = len(batch)
    T_max = max(b["phoneme_ids"].shape[0] for b in batch)
    F_max = max(b["mel"].shape[-1] for b in batch)
    pids = torch.zeros(B, T_max, dtype=torch.long)
    lids = torch.zeros(B, T_max, dtype=torch.long)
    t_lens = torch.zeros(B, dtype=torch.long)
    mels = torch.zeros(B, CFG["n_mels"], F_max, dtype=torch.float32)
    f0s = torch.zeros(B, F_max, dtype=torch.float32)
    es = torch.zeros(B, F_max, dtype=torch.float32)
    m_lens = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        t, f = b["phoneme_ids"].shape[0], b["mel"].shape[-1]
        pids[i, :t] = b["phoneme_ids"]; lids[i, :t] = b["lang_ids"]; t_lens[i] = t
        mels[i, :, :f] = b["mel"]; f0s[i, :f] = b["f0"]; es[i, :f] = b["energy"]; m_lens[i] = f
    return pids, lids, t_lens, mels, f0s, es, m_lens


class VocoderDataset(Dataset):
    def __init__(self, entries, seg_frames: int = 96):
        self.entries = entries
        self.seg_frames = seg_frames
        self.seg_samples = seg_frames * CFG["hop_length"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        d = torch.load(self.entries[idx]["feat_path"], map_location="cpu", weights_only=False)
        wav = decode_wav_tensor(d["wav"])
        if wav.shape[0] < self.seg_samples:
            wav = F.pad(wav, (0, self.seg_samples - wav.shape[0]))
            st = 0
        else:
            st = random.randint(0, wav.shape[0] - self.seg_samples)
        wav_seg = wav[st : st + self.seg_samples]
        with torch.no_grad():
            mel_seg = MEL_EXTRACTOR(wav_seg)[:, :self.seg_frames]
        return mel_seg, wav_seg


train_loader = DataLoader(
    AcousticDataset(DATASET_META["train"]),
    batch_size=CFG["batch_size"], shuffle=True,
    num_workers=CFG["num_workers"], pin_memory=torch.cuda.is_available(),
    collate_fn=acoustic_collate, drop_last=True,
)
val_loader = DataLoader(
    AcousticDataset(DATASET_META["val"]),
    batch_size=min(16, len(DATASET_META["val"])), shuffle=False,
    num_workers=min(2, CFG["num_workers"]), collate_fn=acoustic_collate,
)
print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")


# ==============================================================================
# 7b. 单步耗时分解（--profile-steps N）：定位数据/前向/对齐/反向各占多少
# ==============================================================================
def _run_step_profile():
    if PROFILE_STEPS <= 0:
        return
    exp_name, bb_type, la = EXPERIMENTS[0]
    model = AetherAcousticModel(
        vocab_size=FRONTEND.vocab_size, backbone_type=bb_type, d_model=CFG["d_model"],
        n_mels=CFG["n_mels"], k_left=CFG["k_left"], lookahead=la, swa_window=CFG["swa_window"],
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
    print(f"[Profile] batch={pids.shape[0]} T={pids.shape[1]} F={mels.shape[-1]} | 每步平均耗时:")
    for k in ("h2d", "fwd(含对齐)", "bwd+opt", "e2e"):
        print(f"  {k:12s}: {acc[k] / n * 1000:8.1f} ms")
    with torch.no_grad():
        x_emb = model.phoneme_emb(pids) + model.lang_emb(lids)
        text_mask = torch.arange(pids.shape[1], device=pids.device)[None, :] < t_lens[:, None]
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


def _acc_store(acc, k, dt):
    acc[k] = acc.get(k, 0.0) + dt


_run_step_profile()


# #############################################################################
# Cell 8: 声码器训练
# #############################################################################
VOC_TAG = f"causal_vocos_lite_{CFG['dataset_source']}_{CFG['profile']}"
VOC_STORAGE_DIR = STORAGE_ROOT / "vocoder" / VOC_TAG
VOC_LOCAL_DIR = LOCAL_WORK / "vocoder" / VOC_TAG
VOC_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
VOC_LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def train_or_load_causal_vocoder():
    voc = CausalVocosLite(n_mels=CFG["n_mels"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"]).to(DEVICE)
    disc = LiteVocoderDiscriminator().to(DEVICE)
    opt_g = torch.optim.AdamW(voc.parameters(), lr=CFG["lr_vocoder"], betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=CFG["lr_vocoder"], betas=(0.8, 0.99))

    s1_steps = CFG["vocoder_stage1_steps"]
    s2_steps = CFG["vocoder_stage2_steps"]
    total_steps = s1_steps + s2_steps

    final_storage = VOC_STORAGE_DIR / "final.pt"
    latest_storage = VOC_STORAGE_DIR / "latest.pt"
    done_storage = VOC_STORAGE_DIR / ".done"
    latest_local = VOC_LOCAL_DIR / "latest.pt"
    final_local = VOC_LOCAL_DIR / "final.pt"
    done_local = VOC_LOCAL_DIR / ".done"

    if SKIP_IF_DONE and done_storage.exists() and final_storage.exists():
        print(f"[Resume Done] 直接加载已完结因果声码器: {final_storage}")
        voc.load_state_dict(torch.load(final_storage, map_location=DEVICE, weights_only=True))
        save_progress_status("vocoder", {"status": "done_cached", "step": total_steps, "total_steps": total_steps})
        return voc.eval()

    step = 0
    if latest_storage.exists():
        print(f"[Mid-Run Resume] 检测到未完结声码器断点，正在恢复: {latest_storage}")
        ckpt = torch.load(latest_storage, map_location=DEVICE, weights_only=False)
        voc.load_state_dict(ckpt["voc"])
        disc.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = int(ckpt["step"])
        print(f"  -> 从 step={step}/{total_steps} 无缝继续训练！")

    voc_loader = DataLoader(
        VocoderDataset(DATASET_META["train"], seg_frames=96),
        batch_size=min(16, CFG["batch_size"]), shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    mel_gpu = VocosCompatibleMel(CFG["sample_rate"], CFG["n_fft"], CFG["hop_length"], CFG["n_mels"]).to(DEVICE)

    t0 = time.time()
    last_backup_t = time.time()
    pbar = tqdm(total=total_steps, initial=step, desc="Train CausalVocosLite (Stage1->GAN)")

    def _backup_voc(cur_step: int, is_final: bool = False):
        nonlocal last_backup_t
        payload = {"step": cur_step, "voc": voc.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}
        atomic_torch_save(payload, latest_local)
        sync_file_to_storage(latest_local, latest_storage)
        if is_final:
            atomic_torch_save(voc.state_dict(), final_local)
            sync_file_to_storage(final_local, final_storage)
            atomic_write_text(f"completed_step={cur_step}\n", done_local)
            sync_file_to_storage(done_local, done_storage)
        save_progress_status("vocoder", {"status": "done" if is_final else "running", "step": cur_step, "total_steps": total_steps})
        last_backup_t = time.time()

    try:
        while step < total_steps:
            for mel_seg, wav_gt in voc_loader:
                if step >= total_steps:
                    break
                mel_seg = mel_seg.to(DEVICE, non_blocking=True)
                wav_gt = wav_gt.to(DEVICE, non_blocking=True)
                in_gan = step >= s1_steps
                spec_weight = 5.0 if in_gan else 15.0

                wav_pred = voc(mel_seg)
                L_min = min(wav_pred.shape[-1], wav_gt.shape[-1])
                wav_pred, wav_gt = wav_pred[:, :L_min], wav_gt[:, :L_min]

                if in_gan:
                    opt_d.zero_grad(set_to_none=True)
                    ro, fo, _, _ = disc(wav_gt, wav_pred.detach())
                    loss_d = sum(torch.mean((r - 1.0) ** 2) + torch.mean(f ** 2) for r, f in zip(ro, fo)) / len(ro)
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                    opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                loss_stft = multi_res_stft_loss(wav_pred, wav_gt)
                loss_mel = F.l1_loss(mel_gpu(wav_pred), mel_gpu(wav_gt))
                loss_g = spec_weight * (loss_stft + loss_mel)
                if in_gan:
                    _, fo, rfmaps, ffmaps = disc(wav_gt, wav_pred)
                    loss_adv = sum(torch.mean((f - 1.0) ** 2) for f in fo) / len(fo)
                    loss_fm = sum(F.l1_loss(ff, rf.detach()) for rf_l, ff_l in zip(rfmaps, ffmaps) for rf, ff in zip(rf_l, ff_l)) / (len(rfmaps) * 5)
                    loss_g = loss_g + loss_adv + 2.0 * loss_fm
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(voc.parameters(), 1.0)
                opt_g.step()

                step += 1
                pbar.update(1)
                if step % 200 == 0 or step == total_steps:
                    pbar.set_postfix(stage="GAN" if in_gan else "SPEC", stft=f"{loss_stft.item():.3f}", mel=f"{loss_mel.item():.3f}")

                if _STOP_REQUESTED:
                    _backup_voc(step, is_final=False)
                    print(f"\n[Emergency Backup Saved] CausalVocosLite 已安全保存至 step={step} -> {latest_storage}")
                    raise SystemExit(0)

                need_backup = (step % VOCODER_BACKUP_EVERY_STEPS == 0) or ((time.time() - last_backup_t) >= BACKUP_EVERY_MINUTES * 60.0)
                if need_backup and step < total_steps:
                    _backup_voc(step, is_final=False)
    except KeyboardInterrupt:
        _backup_voc(step, is_final=False)
        print(f"\n[KeyboardInterrupt Backup] CausalVocosLite 已紧急备份至 step={step} -> {latest_storage}")
        raise
    finally:
        pbar.close()

    _backup_voc(step, is_final=True)
    print(f"[Done] CausalVocosLite 训练完成（本次耗时 {(time.time()-t0)/60:.1f} min），已保存至: {VOC_STORAGE_DIR}")
    return voc.eval()


CAUSAL_VOCODER = None
PRETRAINED_VOCOS = None

if CFG["vocoder_mode"] in ("both", "causal_vocos_lite"):
    CAUSAL_VOCODER = train_or_load_causal_vocoder()

if CFG["vocoder_mode"] in ("both", "pretrained_vocos"):
    try:
        from vocos import Vocos
        print("[Load] 加载预训练 charactr/vocos-mel-24khz 作为高质量声码器对照锚点...")
        PRETRAINED_VOCOS = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE).eval()
    except Exception as e:
        print(f"[Warn] 预训练 Vocos 加载失败 ({e})，将仅使用自研 CausalVocosLite。")

# #############################################################################
# Cell 9: 声学主干训练
# #############################################################################
TRAIN_HISTORIES = {}


def train_one_model(exp_name: str, backbone_type: str, lookahead: int):
    tag = f"{exp_name}_{CFG['dataset_source']}_{CFG['profile']}"
    run_local = LOCAL_WORK / "runs" / tag
    run_storage = STORAGE_ROOT / "runs" / tag
    run_local.mkdir(parents=True, exist_ok=True)
    run_storage.mkdir(parents=True, exist_ok=True)
    (run_local / "previews").mkdir(parents=True, exist_ok=True)

    storage_best = run_storage / "best.pt"
    storage_latest = run_storage / "latest.pt"
    storage_hist = run_storage / "history.json"
    storage_done = run_storage / ".done"

    local_best = run_local / "best.pt"
    local_latest = run_local / "latest.pt"
    local_hist = run_local / "history.json"
    local_done = run_local / ".done"

    model = AetherAcousticModel(
        vocab_size=FRONTEND.vocab_size, backbone_type=backbone_type, d_model=CFG["d_model"],
        n_mels=CFG["n_mels"], k_left=CFG["k_left"], lookahead=lookahead, swa_window=CFG["swa_window"],
    ).to(DEVICE)

    if SKIP_IF_DONE and storage_done.exists() and (storage_best.exists() or storage_latest.exists()):
        src_ckpt = storage_best if storage_best.exists() else storage_latest
        print(f"\n[Resume Done] {exp_name} 已存在完结标记 .done，直接加载最优权重: {src_ckpt}")
        ckpt = torch.load(src_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        TRAIN_HISTORIES[exp_name] = json.loads(storage_hist.read_text(encoding="utf-8")) if storage_hist.exists() else {"steps": [], "val_mel": []}
        save_progress_status(exp_name, {"status": "done_cached", "best_val_mel": ckpt.get("best_val_mel")})
        return model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr_acoustic"], betas=(0.9, 0.98), weight_decay=1e-2)
    total_steps = CFG["acoustic_steps"]
    warmup = max(200, int(total_steps * 0.05))

    def lr_lambda(s):
        if s < warmup:
            return float(s + 1) / float(warmup)
        prog = float(s - warmup) / float(max(1, total_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and AMP_DTYPE == torch.float16))

    history = {"steps": [], "train_loss": [], "val_mel": [], "val_dur": [], "elapsed_min": 0.0}
    best_val_mel = float("inf")
    step = 0
    prior_elapsed_min = 0.0

    # 中途精确断点续训：如果存在 latest.pt（但无 .done），无缝恢复完整优化器状态继续跑！
    if storage_latest.exists():
        print(f"\n[Mid-Run Resume] 发现 {exp_name} 未完结断点，正在从 {storage_latest} 精确续训...")
        ckpt = torch.load(storage_latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt["step"])
        best_val_mel = float(ckpt.get("best_val_mel", float("inf")))
        history = ckpt.get("history", history)
        prior_elapsed_min = float(history.get("elapsed_min", 0.0))
        if storage_best.exists() and not _same_path(local_best, storage_best):
            shutil.copy2(storage_best, local_best)
        print(f"  -> 已恢复至 step={step}/{total_steps}, best_val_mel={best_val_mel:.4f}")
    else:
        print(f"\n{'='*72}")
        print(f"开始训练: {exp_name} (backbone={backbone_type}, L={lookahead}, params={count_params_m(model):.2f}M)")
        print(f"{'='*72}")

    t0 = time.time()
    last_backup_t = time.time()
    eval_every = max(300, total_steps // 10)
    preview_text = "你好，请问今天北京的天气怎么样？" if CFG["dataset_source"] != "ljspeech" else "Hello, how is the weather today?"
    pbar = tqdm(total=total_steps, initial=step, desc=f"Train {exp_name}")

    def _save_resumable_latest(cur_step: int):
        nonlocal last_backup_t
        history["elapsed_min"] = round(prior_elapsed_min + (time.time() - t0) / 60.0, 2)
        ckpt_obj = {
            "exp_name": exp_name, "backbone_type": backbone_type, "lookahead": lookahead,
            "step": cur_step, "best_val_mel": best_val_mel, "history": history,
            "cfg": CFG, "stats": STATS, "vocab": FRONTEND.vocab,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        }
        atomic_torch_save(ckpt_obj, local_latest)
        sync_file_to_storage(local_latest, storage_latest)
        atomic_write_text(json.dumps(history, indent=2), local_hist)
        sync_file_to_storage(local_hist, storage_hist)
        save_progress_status(exp_name, {
            "status": "running", "step": cur_step, "total_steps": total_steps,
            "best_val_mel": None if math.isinf(best_val_mel) else round(best_val_mel, 4),
            "elapsed_min": history["elapsed_min"],
        })
        last_backup_t = time.time()

    def _save_preview_wav(cur_step: int):
        voc = PRETRAINED_VOCOS if PRETRAINED_VOCOS is not None else CAUSAL_VOCODER
        if voc is None:
            return
        try:
            pids_p, lids_p = FRONTEND.text_to_ids(preview_text)
            with torch.no_grad():
                mel_n, _, _, _ = model.infer_offline(
                    torch.tensor([pids_p], dtype=torch.long, device=DEVICE),
                    torch.tensor([lids_p], dtype=torch.long, device=DEVICE),
                )
                mel_raw = denorm_mel(mel_n)
                w = voc.decode(mel_raw) if (voc is PRETRAINED_VOCOS) else voc(mel_raw)
                wav_np = np.clip(w.squeeze(0).cpu().float().numpy(), -1.0, 1.0)
            p_loc = run_local / "previews" / f"step_{cur_step:06d}.wav"
            sf.write(p_loc, wav_np, CFG["sample_rate"])
            sync_file_to_storage(p_loc, run_storage / "previews" / p_loc.name)
        except Exception:
            pass

    try:
        while step < total_steps:
            model.train()
            for pids, lids, t_lens, mels, f0s, es, m_lens in train_loader:
                if step >= total_steps:
                    break
                pids = pids.to(DEVICE, non_blocking=True)
                lids = lids.to(DEVICE, non_blocking=True)
                t_lens = t_lens.to(DEVICE, non_blocking=True)
                mels = mels.to(DEVICE, non_blocking=True)
                f0s = f0s.to(DEVICE, non_blocking=True)
                es = es.to(DEVICE, non_blocking=True)
                m_lens = m_lens.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                    out = model.forward_train(pids, lids, t_lens, mels, f0s, es, m_lens, step_ratio=step / max(total_steps, 1))
                    loss = out["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step += 1
                pbar.update(1)
                if step % 100 == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}", mel=f"{out['mel'].item():.3f}",
                        dur=f"{out['dur'].item():.3f}", aln=f"{out['align'].item():.3f}",
                        bin=f"{out['bin'].item():.3f}", bos=f"{out['bos_ratio'].item():.2f}",
                    )

                if _STOP_REQUESTED:
                    _save_resumable_latest(step)
                    print(f"\n[Emergency Backup Saved] {exp_name} 已安全保存至 step={step} -> {storage_latest}")
                    raise SystemExit(0)

                # 定期验证 + 保存 best.pt + 导出训练中试听预览
                if step % eval_every == 0 or step == total_steps:
                    model.eval()
                    v_mel, v_dur, v_bos, v_cnt = 0.0, 0.0, 0.0, 0
                    with torch.no_grad():
                        for vp, vl, vtl, vm, vf0, ve, vml in val_loader:
                            vo = model.forward_train(
                                vp.to(DEVICE), vl.to(DEVICE), vtl.to(DEVICE),
                                vm.to(DEVICE), vf0.to(DEVICE), ve.to(DEVICE), vml.to(DEVICE),
                                step_ratio=1.0,
                            )
                            v_mel += vo["mel"].item(); v_dur += vo["dur"].item(); v_bos += vo["bos_ratio"].item(); v_cnt += 1
                    v_mel /= max(v_cnt, 1); v_dur /= max(v_cnt, 1); v_bos /= max(v_cnt, 1)
                    history["steps"].append(step)
                    history["train_loss"].append(round(loss.item(), 4))
                    history["val_mel"].append(round(v_mel, 4))
                    history["val_dur"].append(round(v_dur, 4))
                    history.setdefault("val_bos_ratio", []).append(round(v_bos, 3))

                    if not math.isnan(v_mel) and v_mel < best_val_mel:
                        best_val_mel = v_mel
                        best_payload = {
                            "exp_name": exp_name, "backbone_type": backbone_type, "lookahead": lookahead,
                            "step": step, "best_val_mel": best_val_mel, "cfg": CFG, "stats": STATS,
                            "vocab": FRONTEND.vocab, "model": model.state_dict(),
                        }
                        atomic_torch_save(best_payload, local_best)
                        sync_file_to_storage(local_best, storage_best)
                    _save_preview_wav(step)
                    _save_resumable_latest(step)
                    model.train()

                # 额外高频备份触发（按步数或按时间）
                elif (step % BACKUP_EVERY_STEPS == 0) or ((time.time() - last_backup_t) >= BACKUP_EVERY_MINUTES * 60.0):
                    _save_resumable_latest(step)

                # 不可覆盖里程碑权重快照
                if SAVE_MILESTONE_EVERY > 0 and (step % SAVE_MILESTONE_EVERY == 0) and step < total_steps:
                    ms_local = run_local / f"milestone_step_{step:06d}.pt"
                    atomic_torch_save({"step": step, "model": model.state_dict()}, ms_local)
                    sync_file_to_storage(ms_local, run_storage / ms_local.name)
    except KeyboardInterrupt:
        _save_resumable_latest(step)
        print(f"\n[KeyboardInterrupt Backup] {exp_name} 已紧急备份至 step={step} -> {storage_latest}")
        raise
    finally:
        pbar.close()

    # 完结后加载 best.pt（若不存在则回退 latest.pt）并导出动态 INT8 包，原子写入 .done 标记
    best_src = next((p for p in (local_best, storage_best, local_latest, storage_latest) if p.exists()), None)
    if best_src is not None:
        ckpt = torch.load(best_src, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if not storage_best.exists():
            best_payload = {
                "exp_name": exp_name, "backbone_type": backbone_type, "lookahead": lookahead,
                "step": step, "best_val_mel": best_val_mel, "cfg": CFG, "stats": STATS,
                "vocab": FRONTEND.vocab, "model": model.state_dict(),
            }
            atomic_torch_save(best_payload, local_best)
            sync_file_to_storage(local_best, storage_best)

    model_cpu = model.to("cpu").eval()
    try:
        q_model = torch.ao.quantization.quantize_dynamic(model_cpu, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        local_int8 = run_local / "model_int8.pt"
        atomic_torch_save(q_model.state_dict(), local_int8)
        sync_file_to_storage(local_int8, run_storage / "model_int8.pt")
    except Exception as e:
        print(f"[Warn] INT8 dynamic export skipped for {exp_name}: {e}")

    atomic_write_text(f"completed_step={step}\nbest_val_mel={best_val_mel:.4f}\n", local_done)
    sync_file_to_storage(local_done, storage_done)
    save_progress_status(exp_name, {
        "status": "done", "step": step, "total_steps": total_steps,
        "best_val_mel": None if math.isinf(best_val_mel) else round(best_val_mel, 4),
        "elapsed_min": history["elapsed_min"],
    })
    model = model.to(DEVICE).eval()
    TRAIN_HISTORIES[exp_name] = history
    print(f"[Done] {exp_name} | Best Val Mel L1: {best_val_mel:.4f} | Total Time: {history['elapsed_min']} min | 保存位置: {run_storage}")
    return model


TRAINED_MODELS = {}
for exp_name, bb_type, la in EXPERIMENTS:
    TRAINED_MODELS[exp_name] = train_one_model(exp_name, bb_type, la)

# #############################################################################
# Cell 10: 评测音频合成
# #############################################################################
GOLDEN_ZH = [
    ("zh_qa_01", "你好，请问今天北京的天气怎么样？"),
    ("zh_qa_02", "好的，已经帮你把明早七点的闹钟设置好了。"),
    ("zh_poly_01", "中国人民银行行长今天上午发表了重要讲话。"),
    ("zh_poly_02", "这件事情非常重，我们不要重复同样的错误。"),
    ("zh_sandhi_01", "你好，我想去展览馆买把好雨伞。"),
    ("zh_tn_01", "今年是2026年，同比增长了18.5%。"),
    ("zh_short_01", "好的，没问题。"),
    ("zh_short_02", "收到，马上处理。"),
    ("zh_long_01", "在嵌入式设备上实现端侧语音合成时，需要在自然度、首包延迟和纯中央处理器算力占用之间找到一个非常精准的平衡点。"),
    ("zh_cs_01", "帮我把 Wi-Fi 切换到 5G，顺便查一下今天的 API latency。"),
]
GOLDEN_EN = [
    ("en_qa_01", "Hello, how is the weather in San Francisco today?"),
    ("en_qa_02", "Sure, I have set your alarm for seven thirty tomorrow morning."),
    ("en_short_01", "Got it, no problem."),
    ("en_short_02", "Sure thing."),
    ("en_tech_01", "Streaming text to speech requires low latency and consistent speaker identity."),
    ("en_long_01", "Running neural speech synthesis directly on an embedded Linux processor eliminates cloud round trips while protecting user privacy."),
]


def build_eval_suite():
    suite = []
    for e in DATASET_META["val"][: CFG["eval_max_utts"]]:
        suite.append({"utt_id": f"val_{e['utt_id']}", "text": e["text"], "feat_path": e["feat_path"], "is_val": True})
    extras = GOLDEN_ZH if CFG["dataset_source"] == "bznsyp" else (GOLDEN_EN if CFG["dataset_source"] == "ljspeech" else GOLDEN_ZH + GOLDEN_EN)
    for uid, txt in extras:
        suite.append({"utt_id": uid, "text": txt, "feat_path": None, "is_val": False})
    return suite


EVAL_SUITE = build_eval_suite()
print(f"评测集构建完成：共 {len(EVAL_SUITE)} 句（含真声参考 Val: {sum(x['is_val'] for x in EVAL_SUITE)} 句）")


@torch.no_grad()
def synthesize_offline(model: AetherAcousticModel, text: str, vocoder, use_pretrained_vocos: bool = False):
    pids, lids = FRONTEND.text_to_ids(text)
    mel_norm, _, _, _ = model.infer_offline(
        torch.tensor([pids], dtype=torch.long, device=DEVICE),
        torch.tensor([lids], dtype=torch.long, device=DEVICE),
    )
    mel = denorm_mel(mel_norm)
    wav = PRETRAINED_VOCOS.decode(mel) if (use_pretrained_vocos and PRETRAINED_VOCOS is not None) else vocoder(mel)
    return wav.squeeze(0).cpu().float().numpy(), mel.squeeze(0).cpu().float().numpy()


@torch.no_grad()
def synthesize_streaming(model: AetherAcousticModel, text: str, vocoder: CausalVocosLite):
    pids, lids = FRONTEND.text_to_ids(text)
    pt = torch.tensor([pids], dtype=torch.long, device=DEVICE)
    lt = torch.tensor([lids], dtype=torch.long, device=DEVICE)
    T = pt.shape[1]
    ast_state = model.init_stream_state(1, device=DEVICE, dtype=torch.float32)
    voc_state = vocoder.init_state(1, device=DEVICE, dtype=torch.float32)
    mel_buf = torch.zeros(1, CFG["n_mels"], 0, device=DEVICE, dtype=torch.float32)
    wav_chunks, mel_chunks, seam_positions = [], [], []
    cur_samples = 0
    t_step, m_step = CFG["stream_text_chunk"], CFG["stream_mel_chunk"]

    for i in range(0, T, t_step):
        is_last = (i + t_step >= T)
        mc, ast_state = model.infer_stream_step(pt[:, i:i+t_step], lt[:, i:i+t_step], ast_state, is_final=is_last)
        if mc.shape[-1] > 0:
            mc_d = denorm_mel(mc)
            mel_chunks.append(mc_d)
            mel_buf = torch.cat([mel_buf, mc_d], dim=-1)
        while mel_buf.shape[-1] >= m_step:
            wc, voc_state = vocoder.forward_step(mel_buf[:, :, :m_step], voc_state, is_final=False)
            mel_buf = mel_buf[:, :, m_step:]
            if wc.shape[-1] > 0:
                if cur_samples > 0:
                    seam_positions.append(cur_samples)
                wav_chunks.append(wc.squeeze(0).cpu().float().numpy())
                cur_samples += wc.shape[-1]
        if is_last:
            wc, voc_state = vocoder.forward_step(mel_buf, voc_state, is_final=True)
            if wc.shape[-1] > 0:
                if cur_samples > 0:
                    seam_positions.append(cur_samples)
                wav_chunks.append(wc.squeeze(0).cpu().float().numpy())
                cur_samples += wc.shape[-1]

    wav = np.concatenate(wav_chunks) if wav_chunks else np.zeros(2400, dtype=np.float32)
    mel_all = torch.cat(mel_chunks, dim=-1).squeeze(0).cpu().float().numpy() if mel_chunks else np.zeros((CFG["n_mels"], 1))
    return wav, mel_all, seam_positions


EVAL_LOCAL_DIR = LOCAL_WORK / "eval_samples" / f"{CFG['dataset_source']}_{CFG['profile']}"
EVAL_STORAGE_DIR = STORAGE_ROOT / "eval_samples" / f"{CFG['dataset_source']}_{CFG['profile']}"
EVAL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

gt_dir = EVAL_LOCAL_DIR / "00_ground_truth"
gt_dir.mkdir(parents=True, exist_ok=True)
for item in EVAL_SUITE:
    if item["feat_path"] is not None:
        d = torch.load(item["feat_path"], map_location="cpu", weights_only=False)
        sf.write(gt_dir / f"{item['utt_id']}.wav", decode_wav_tensor(d["wav"]).numpy(), CFG["sample_rate"])
sync_dir_to_storage(gt_dir, EVAL_STORAGE_DIR / "00_ground_truth")

GENERATED_RECORDS = {}
use_pre_for_offline = (CAUSAL_VOCODER is None)

for exp_name, model in TRAINED_MODELS.items():
    exp_out_off = EVAL_LOCAL_DIR / exp_name / "offline"
    exp_out_str = EVAL_LOCAL_DIR / exp_name / "streaming"
    exp_out_pre = EVAL_LOCAL_DIR / exp_name / "pretrained_vocos"
    exp_out_off.mkdir(parents=True, exist_ok=True)
    if model.is_streaming and CAUSAL_VOCODER is not None:
        exp_out_str.mkdir(parents=True, exist_ok=True)
    if PRETRAINED_VOCOS is not None:
        exp_out_pre.mkdir(parents=True, exist_ok=True)

    recs = []
    for item in tqdm(EVAL_SUITE, desc=f"Synthesize {exp_name}"):
        wav_off, _ = synthesize_offline(
            model, item["text"],
            vocoder=CAUSAL_VOCODER if not use_pre_for_offline else PRETRAINED_VOCOS,
            use_pretrained_vocos=use_pre_for_offline,
        )
        p_off = exp_out_off / f"{item['utt_id']}.wav"
        sf.write(p_off, np.clip(wav_off, -1.0, 1.0), CFG["sample_rate"])

        p_pre = None
        if PRETRAINED_VOCOS is not None and not use_pre_for_offline:
            wav_pre, _ = synthesize_offline(model, item["text"], vocoder=None, use_pretrained_vocos=True)
            p_pre = exp_out_pre / f"{item['utt_id']}.wav"
            sf.write(p_pre, np.clip(wav_pre, -1.0, 1.0), CFG["sample_rate"])

        p_str, seams, stream_diff = None, [], 0.0
        if model.is_streaming and CAUSAL_VOCODER is not None:
            wav_str, _, seams = synthesize_streaming(model, item["text"], CAUSAL_VOCODER)
            p_str = exp_out_str / f"{item['utt_id']}.wav"
            sf.write(p_str, np.clip(wav_str, -1.0, 1.0), CFG["sample_rate"])
            L_min = min(len(wav_off), len(wav_str))
            stream_diff = float(np.max(np.abs(wav_off[:L_min] - wav_str[:L_min]))) if L_min > 0 else 0.0

        recs.append({
            "utt_id": item["utt_id"], "text": item["text"], "is_val": item["is_val"],
            "gt_wav": str(gt_dir / f"{item['utt_id']}.wav") if item["is_val"] else None,
            "off_wav": str(p_off), "str_wav": str(p_str) if p_str else None,
            "pre_wav": str(p_pre) if p_pre else None,
            "seams": seams, "stream_wav_max_diff": stream_diff,
        })
    GENERATED_RECORDS[exp_name] = recs
    # 每合成完一个模型立即同步，避免中途断连前功尽弃
    sync_dir_to_storage(EVAL_LOCAL_DIR / exp_name, EVAL_STORAGE_DIR / exp_name)

save_progress_status("synthesis", {"status": "done", "models": list(GENERATED_RECORDS.keys()), "utts_per_model": len(EVAL_SUITE)})
print(f"[Saved] 全部生成音频已归档至: {EVAL_STORAGE_DIR}")

# #############################################################################
# Cell 11: 自动打分与 CPU 基准
# #############################################################################
import pandas as pd
from jiwer import wer as calc_wer

UTMOS_MODEL = None
try:
    print("[Eval] 加载 UTMOS (utmos22_strong)...")
    UTMOS_MODEL = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEVICE).eval()
except Exception as e:
    print(f"[Warn] UTMOS unavailable: {e}")

SQUIM_MODEL = None
try:
    print("[Eval] 加载 TorchAudio SQUIM_OBJECTIVE...")
    SQUIM_MODEL = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(DEVICE).eval()
except Exception as e:
    print(f"[Warn] SQUIM unavailable: {e}")

SPK_ENCODER = None
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    print("[Eval] 加载 Resemblyzer 声纹编码器...")
    SPK_ENCODER = VoiceEncoder(device=str(DEVICE))
except Exception as e:
    print(f"[Warn] Resemblyzer unavailable: {e}")

WHISPER_MODEL = None
try:
    from faster_whisper import WhisperModel
    w_name = "small" if CFG["profile"] == "quick_smoke" else "large-v3-turbo"
    print(f"[Eval] 加载 Faster-Whisper ({w_name})...")
    WHISPER_MODEL = WhisperModel(w_name, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
except Exception as e:
    print(f"[Warn] Faster-Whisper unavailable: {e}")


def compute_mixed_error_rate(ref: str, hyp: str) -> float:
    def tokenize(s):
        s = unicodedata.normalize("NFKC", s).lower()
        return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", s)
    r_toks, h_toks = tokenize(ref), tokenize(hyp)
    return 0.0 if not r_toks else float(calc_wer(" ".join(r_toks), " ".join(h_toks)))


@torch.no_grad()
def score_utmos(wav_np: np.ndarray, sr: int = 24000) -> float:
    if UTMOS_MODEL is None:
        return float("nan")
    w16 = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
    return float(UTMOS_MODEL(torch.from_numpy(w16).unsqueeze(0).to(DEVICE), 16000).item())


@torch.no_grad()
def score_squim(wav_np: np.ndarray, sr: int = 24000):
    if SQUIM_MODEL is None:
        return float("nan"), float("nan"), float("nan")
    w16 = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
    stoi, pesq, sisdr = SQUIM_MODEL(torch.from_numpy(w16).unsqueeze(0).to(DEVICE))
    return float(pesq.item()), float(sisdr.item()), float(stoi.item())


def hf_2_8k_ratio(wav_np: np.ndarray, sr: int = 24000) -> float:
    spec = np.abs(np.fft.rfft(wav_np)) ** 2
    freqs = np.fft.rfftfreq(len(wav_np), 1.0 / sr)
    band = (freqs >= 2000.0) & (freqs <= 8000.0)
    return float(spec[band].sum() / (spec.sum() + 1e-8))


def detect_seam_click(wav_np: np.ndarray, seams: list, sr: int = 24000) -> float:
    if not seams:
        return 0.0
    diff = np.abs(np.diff(wav_np))
    win = int(0.005 * sr)
    global_rms = float(np.sqrt(np.mean(diff ** 2) + 1e-10))
    clicks = 0
    for pos in seams:
        lo, hi = max(0, pos - win), min(len(diff), pos + win)
        if hi <= lo:
            continue
        local_max = float(np.max(diff[lo:hi]))
        ref_lo, ref_hi = max(0, pos - 4 * win), min(len(diff), pos + 4 * win)
        local_rms = float(np.sqrt(np.mean(diff[ref_lo:ref_hi] ** 2) + 1e-10))
        if local_max > max(12.0 * local_rms, 15.0 * global_rms):
            clicks += 1
    return float(clicks) / len(seams)


def compute_asr_error(wav_or_path, ref_text: str, sr: int = 24000) -> float:
    if WHISPER_MODEL is None:
        return float("nan")
    try:
        if isinstance(wav_or_path, (str, Path)):
            wav_np, sr = sf.read(str(wav_or_path), dtype="float32")
        else:
            wav_np = np.asarray(wav_or_path, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        if len(wav_np) == 0:
            return 1.0
        # 直接传入 16 kHz float32 ndarray，彻底绕开 faster-whisper 内部 PyAV av.open(..., metadata_errors="ignore") 在新版 av/Python 3.13 上的 TypeError
        w16 = librosa.resample(wav_np, orig_sr=sr, target_sr=16000) if sr != 16000 else wav_np
        w16 = np.ascontiguousarray(w16, dtype=np.float32)
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", ref_text) else "en"
        segs, _ = WHISPER_MODEL.transcribe(w16, language=lang, beam_size=1)
        return compute_mixed_error_rate(ref_text, "".join(s.text for s in segs))
    except Exception as e:
        print(f"[Warn] ASR scoring failed ({e}), returning NaN.")
        return float("nan")


def _safe_nanmean(vals) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if valid.size > 0 else float("nan")


GT_REF_EMBS, GT_HF_RATIOS, GT_UTMOS_LIST = [], [], []
for item in EVAL_SUITE:
    if item["is_val"]:
        p = EVAL_LOCAL_DIR / "00_ground_truth" / f"{item['utt_id']}.wav"
        w, _ = sf.read(str(p), dtype="float32")
        GT_HF_RATIOS.append(hf_2_8k_ratio(w))
        GT_UTMOS_LIST.append(score_utmos(w))
        if SPK_ENCODER is not None:
            GT_REF_EMBS.append(SPK_ENCODER.embed_utterance(preprocess_wav(w, source_sr=CFG["sample_rate"])))

GT_SPK_CENTROID = np.mean(GT_REF_EMBS, axis=0) if GT_REF_EMBS else None
if GT_SPK_CENTROID is not None:
    GT_SPK_CENTROID /= (np.linalg.norm(GT_SPK_CENTROID) + 1e-8)
GT_HF_MEAN = float(np.mean(GT_HF_RATIOS)) if GT_HF_RATIOS else 0.15
GT_UTMOS_MEAN = _safe_nanmean(GT_UTMOS_LIST)
print(f"[Gold Anchor] Ground-Truth UTMOS: {GT_UTMOS_MEAN:.3f} | 2-8kHz Ratio: {GT_HF_MEAN:.4f}")


@torch.no_grad()
def benchmark_cpu_2thread(model: AetherAcousticModel, vocoder: CausalVocosLite, sample_text: str):
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    m_cpu = model.to("cpu").eval()
    v_cpu = vocoder.to("cpu").eval() if vocoder is not None else None
    pids, lids = FRONTEND.text_to_ids(sample_text)
    pt, lt = torch.tensor([pids], dtype=torch.long), torch.tensor([lids], dtype=torch.long)
    T = pt.shape[1]

    def _run_once():
        if m_cpu.is_streaming and v_cpu is not None:
            t_start, ttfa_ms, total_samples = time.perf_counter(), None, 0
            ast_st = m_cpu.init_stream_state(1, device=torch.device("cpu"), dtype=torch.float32)
            voc_st = v_cpu.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
            mel_buf = torch.zeros(1, CFG["n_mels"], 0)
            t_step, m_step = CFG["stream_text_chunk"], CFG["stream_mel_chunk"]
            for i in range(0, T, t_step):
                is_last = (i + t_step >= T)
                mc, ast_st = m_cpu.infer_stream_step(pt[:, i:i+t_step], lt[:, i:i+t_step], ast_st, is_final=is_last)
                if mc.shape[-1] > 0:
                    mel_buf = torch.cat([mel_buf, denorm_mel(mc)], dim=-1)
                while mel_buf.shape[-1] >= m_step:
                    wc, voc_st = v_cpu.forward_step(mel_buf[:, :, :m_step], voc_st, is_final=False)
                    mel_buf = mel_buf[:, :, m_step:]
                    total_samples += wc.shape[-1]
                    if ttfa_ms is None and wc.shape[-1] > 0:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000.0
                if is_last:
                    wc, voc_st = v_cpu.forward_step(mel_buf, voc_st, is_final=True)
                    total_samples += wc.shape[-1]
                    if ttfa_ms is None and wc.shape[-1] > 0:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000.0
            elapsed = time.perf_counter() - t_start
            return elapsed / max(total_samples / CFG["sample_rate"], 1e-3), (ttfa_ms or 0.0)
        t_start = time.perf_counter()
        mel = denorm_mel(m_cpu.infer_offline(pt, lt)[0])
        audio_sec = max((v_cpu(mel).shape[-1] if v_cpu is not None else mel.shape[-1] * CFG["hop_length"]) / CFG["sample_rate"], 1e-3)
        elapsed = time.perf_counter() - t_start
        return elapsed / audio_sec, elapsed * 1000.0

    # 先预热 1 遍真实推理路径，再跑 3 遍取中位数，消除云端 vCPU 瞬时抖动
    _run_once()
    runs = [_run_once() for _ in range(3)]
    rtf = float(np.median([x[0] for x in runs]))
    ttfa_ms = float(np.median([x[1] for x in runs]))

    model.to(DEVICE)
    if vocoder is not None:
        vocoder.to(DEVICE)
    torch.set_num_threads(prev_threads)
    return round(rtf, 4), round(ttfa_ms, 1)


all_sample_rows, summary_rows = [], []
bench_text = GOLDEN_ZH[-2][1] if CFG["dataset_source"] != "ljspeech" else GOLDEN_EN[-1][1]

for exp_name, model in TRAINED_MODELS.items():
    recs = GENERATED_RECORDS[exp_name]
    utmos_s, pesq_s, sisdr_s, stoi_s, mer_s, sim_s, hf_diff_s, click_s, sdiff_s, utmos_pre_s = [], [], [], [], [], [], [], [], [], []
    for r in tqdm(recs, desc=f"Scoring {exp_name}"):
        eval_wav_path = r["str_wav"] if r["str_wav"] else r["off_wav"]
        wav, wav_sr = sf.read(eval_wav_path, dtype="float32")
        u = score_utmos(wav, sr=wav_sr)
        pesq, sisdr, stoi = score_squim(wav, sr=wav_sr)
        mer = compute_asr_error(wav, r["text"], sr=wav_sr)
        hfd = abs(hf_2_8k_ratio(wav, sr=wav_sr) - GT_HF_MEAN)
        clk = detect_seam_click(wav, r["seams"], sr=wav_sr) if r["str_wav"] else 0.0
        sim = float("nan")
        if SPK_ENCODER is not None and GT_SPK_CENTROID is not None:
            emb = SPK_ENCODER.embed_utterance(preprocess_wav(wav, source_sr=wav_sr))
            sim = float(np.dot(emb / (np.linalg.norm(emb) + 1e-8), GT_SPK_CENTROID))
        u_pre = float("nan")
        if r["pre_wav"]:
            w_pre, w_pre_sr = sf.read(r["pre_wav"], dtype="float32")
            u_pre = score_utmos(w_pre, sr=w_pre_sr)
            utmos_pre_s.append(u_pre)
        utmos_s.append(u); pesq_s.append(pesq); sisdr_s.append(sisdr); stoi_s.append(stoi)
        mer_s.append(mer); sim_s.append(sim); hf_diff_s.append(hfd); click_s.append(clk); sdiff_s.append(r["stream_wav_max_diff"])
        all_sample_rows.append({"model": exp_name, "utt_id": r["utt_id"], "text": r["text"], "UTMOS": u, "UTMOS_PreVocos": u_pre, "PESQ": pesq, "SI_SDR": sisdr, "MER": mer, "SIM": sim, "HF_2_8k_Diff": hfd, "Seam_Click": clk})

    rtf_2t, ttfa_ms = benchmark_cpu_2thread(model, CAUSAL_VOCODER, bench_text)
    infer_params_m = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("aligner.")) / 1e6
    int8_path = STORAGE_ROOT / "runs" / f"{exp_name}_{CFG['dataset_source']}_{CFG['profile']}" / "model_int8.pt"
    size_mb = round(int8_path.stat().st_size / 1e6, 2) if int8_path.exists() else round(infer_params_m * 4.0, 2)

    mean_utmos = _safe_nanmean(utmos_s)
    mean_utmos_pre = _safe_nanmean(utmos_pre_s)
    mean_mer, mean_sim, mean_pesq, mean_click = _safe_nanmean(mer_s), _safe_nanmean(sim_s), _safe_nanmean(pesq_s), float(np.mean(click_s))

    s_nat = np.clip((mean_utmos - 1.5) / 3.0, 0.0, 1.0) * 35.0 if not np.isnan(mean_utmos) else 20.0
    s_int = np.clip(1.0 - mean_mer * 3.0, 0.0, 1.0) * 30.0 if not np.isnan(mean_mer) else 20.0
    s_sim = np.clip((mean_sim - 0.5) / 0.45, 0.0, 1.0) * 15.0 if not np.isnan(mean_sim) else 10.0
    s_lat = np.clip(1.0 - ttfa_ms / 500.0, 0.0, 1.0) * 10.0 + np.clip(1.0 - rtf_2t / 0.5, 0.0, 1.0) * 10.0
    composite = round(float(s_nat + s_int + s_sim + s_lat - 20.0 * mean_click), 2)

    summary_rows.append({
        "Model": exp_name, "Streaming": model.is_streaming, "Lookahead": model.lookahead,
        "Params_M": round(infer_params_m, 2), "Size_MB": size_mb,
        "UTMOS↑": round(mean_utmos, 3), "UTMOS_PreVocos↑": round(mean_utmos_pre, 3) if not np.isnan(mean_utmos_pre) else None,
        "MER/CER↓": round(mean_mer, 4), "SIM↑": round(mean_sim, 3), "SQUIM_PESQ↑": round(mean_pesq, 3),
        "HF_2_8k_Diff↓": round(float(np.mean(hf_diff_s)), 4), "Seam_Click↓": round(mean_click, 4),
        "Stream_MaxDiff": f"{max(sdiff_s):.1e}", "CPU2T_RTF↓": rtf_2t, "CPU2T_TTFA_ms↓": ttfa_ms, "Composite_Score↑": composite,
    })

LEADERBOARD = pd.DataFrame(summary_rows).sort_values("Composite_Score↑", ascending=False).reset_index(drop=True)
PER_SAMPLE_DF = pd.DataFrame(all_sample_rows)

report_local = LOCAL_WORK / "reports" / f"{CFG['dataset_source']}_{CFG['profile']}"
report_storage = STORAGE_ROOT / "reports" / f"{CFG['dataset_source']}_{CFG['profile']}"
report_local.mkdir(parents=True, exist_ok=True)
report_storage.mkdir(parents=True, exist_ok=True)

LEADERBOARD.to_csv(report_local / "leaderboard.csv", index=False)
LEADERBOARD.to_json(report_local / "leaderboard.json", orient="records", force_ascii=False, indent=2)
PER_SAMPLE_DF.to_csv(report_local / "per_sample_scores.csv", index=False)
sync_dir_to_storage(report_local, report_storage)

print("\n================ AetherTTS Phase 1 综合评测排行榜 ================")
_in_nb = False
try:
    from IPython import get_ipython
    from IPython.display import display
    if get_ipython() is not None:
        display(LEADERBOARD)
        _in_nb = True
except Exception:
    pass
if not _in_nb:
    print(LEADERBOARD.to_string(index=False))
save_progress_status("evaluation", {"status": "done", "top_model": str(LEADERBOARD.iloc[0]["Model"]) if len(LEADERBOARD) else None})
print(f"\n[Saved] 评测报告已保存至: {report_storage}")

# #############################################################################
# Cell 12: 可视化与盲听表
# #############################################################################
import matplotlib
if not IS_COLAB and os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for exp_name, hist in TRAIN_HISTORIES.items():
    if hist.get("steps"):
        ax.plot(hist["steps"], hist["val_mel"], marker="o", label=exp_name)
ax.set_title("Validation Mel L1 Convergence (Lower is Better)")
ax.set_xlabel("Steps"); ax.set_ylabel("Val Mel L1"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
x = np.arange(len(LEADERBOARD))
ax.bar(x - 0.18, LEADERBOARD["UTMOS↑"].fillna(0), width=0.36, label="UTMOS (CausalVocosLite)")
if LEADERBOARD["UTMOS_PreVocos↑"].notna().any():
    ax.bar(x + 0.18, LEADERBOARD["UTMOS_PreVocos↑"].fillna(0), width=0.36, label="UTMOS (Pretrained Vocos)")
if not np.isnan(GT_UTMOS_MEAN):
    ax.axhline(GT_UTMOS_MEAN, color="red", linestyle="--", label=f"Gold GT ({GT_UTMOS_MEAN:.2f})")
ax.set_xticks(x); ax.set_xticklabels(LEADERBOARD["Model"], rotation=25, ha="right", fontsize=8)
ax.set_title("Naturalness (UTMOS)"); ax.set_ylim(0, 5.0); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

ax = axes[2]
ax.bar(x, LEADERBOARD["CPU2T_TTFA_ms↓"], color="#4C78A8", alpha=0.85, label="TTFA (ms)")
ax.axhline(150.0, color="green", linestyle="--", label="Target TTFA (150ms)")
ax.set_xticks(x); ax.set_xticklabels(LEADERBOARD["Model"], rotation=25, ha="right", fontsize=8)
ax.set_title("CPU 2-Thread First-Audio Latency (ms)"); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
chart_local = report_local / "comparison_charts.png"
plt.savefig(chart_local, dpi=150)
sync_file_to_storage(chart_local, report_storage / "comparison_charts.png")
if IS_COLAB or os.environ.get("DISPLAY", ""):
    plt.show()
else:
    plt.close(fig)

# 盲听打分表（在 Notebook 界面下渲染试听播放器；自建无界面服务器直接导出 CSV）
blind_rows = []
rng_b = random.Random(CFG["seed"] + 99)
model_names = list(TRAINED_MODELS.keys())
preview_utts = [u for u in EVAL_SUITE if not u["is_val"]][:5] or EVAL_SUITE[:5]

interactive_nb = False
try:
    from IPython import get_ipython
    from IPython.display import Audio, display, HTML
    interactive_nb = get_ipython() is not None
except Exception:
    pass

for u in preview_utts:
    if interactive_nb:
        display(HTML(f"<h4>[{u['utt_id']}] {u['text']}</h4>"))
    shuffled = list(model_names)
    rng_b.shuffle(shuffled)
    for label_idx, m_name in enumerate(shuffled):
        rec = next(r for r in GENERATED_RECORDS[m_name] if r["utt_id"] == u["utt_id"])
        wav_p = rec["str_wav"] if rec["str_wav"] else rec["off_wav"]
        tag = f"Candidate_{chr(65 + label_idx)}"
        blind_rows.append({"utt_id": u["utt_id"], "text": u["text"], "blind_tag": tag, "true_model": m_name, "wav_path": wav_p, "CMOS_score": "", "defect_tags": ""})
        if interactive_nb:
            print(f"  {tag}:")
            display(Audio(filename=wav_p, rate=CFG["sample_rate"]))

blind_df = pd.DataFrame(blind_rows)
blind_local = report_local / "blind_ab_listening_sheet.csv"
blind_df.to_csv(blind_local, index=False)
sync_file_to_storage(blind_local, report_storage / "blind_ab_listening_sheet.csv")
save_progress_status("all_complete", {"status": "done", "report_dir": str(report_storage)})
print(f"[Saved] 图表与盲听打分表已保存至: {report_storage}")

if IS_COLAB and AUTO_UNASSIGN_COLAB:
    print("[Colab] 全部任务已完成并同步至 Google Drive，10 秒后自动释放 GPU 运行时...")
    time.sleep(10)
    from google.colab import runtime
    runtime.unassign()

# #############################################################################
# Cell 13: 可选教师造数（默认跳过）
# #############################################################################
RUN_TEACHER_SYNTH = os.environ.get("AETHER_TEACHER_SYNTH", "0") == "1"

if RUN_TEACHER_SYNTH:
    OUT_CUSTOM_DIR = STORAGE_ROOT / "datasets" / "custom_bilingual"
    (OUT_CUSTOM_DIR / "wavs").mkdir(parents=True, exist_ok=True)
    print(f"教师造数输出目录已就绪: {OUT_CUSTOM_DIR}")
    print("参考 docs/03_teacher_distillation_pipeline.md：放入合规干声 ref.wav -> 用 Qwen3-TTS / CosyVoice 批量合成 20-25h -> ASR+SIM+UTMOS 四重清洗 -> 设置 DATASET_SOURCE='custom_bilingual' 开训。")
else:
    print("当前跳过教师离线造数。")

print('[All Done] 独立 Python 实验全部执行完毕！', flush=True)
