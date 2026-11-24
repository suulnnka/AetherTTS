#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a5) — 纯模型与前端基础设施定义库（import 零副作用）。

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
    "stream_mel_first_chunk": 6,
    "stream_mel_chunk": 16,
    "frame_lookahead": int(os.environ.get("AETHER_FRAME_LOOKAHEAD", "2")),
    "vocoder_lookahead": int(os.environ.get("AETHER_VOCODER_LOOKAHEAD", "2")),
    "profile": os.environ.get("AETHER_PROFILE", "standard_phase1"),
    "dataset_source": os.environ.get("AETHER_DATASET", "bznsyp"),
}

# #############################################################################
# Cell 3: 基础设施（Mel/F0+UV/中英音系与韵律前端）
# #############################################################################
import gc
import re
import copy
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

_EN_G2P = None
_EN_G2P_INIT_TRIED = False


def _get_en_g2p():
    global _EN_G2P, _EN_G2P_INIT_TRIED
    if _EN_G2P_INIT_TRIED:
        return _EN_G2P
    _EN_G2P_INIT_TRIED = True
    try:
        from g2p_en import G2p
        import nltk
        for _res_path, _res_name in [
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/cmudict", "cmudict"),
        ]:
            try:
                nltk.data.find(_res_path)
            except LookupError:
                try:
                    nltk.download(_res_name, quiet=True)
                except Exception:
                    pass
        _EN_G2P = G2p()
    except Exception as e:
        print(f"[Warn] g2p_en unavailable ({e}), falling back to character-level EN G2P.")
        _EN_G2P = None
    return _EN_G2P


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
    """24kHz 重采样 + 首尾静音裁剪 + 有声段门控 RMS 响度归一化（消除句中长逗号停顿拉低均值造成的音量跳变）。"""
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
    _, idx = librosa.effects.trim(wav, top_db=35, frame_length=1024, hop_length=256)
    pad = int(0.06 * target_sr)
    start = max(0, idx[0] - pad)
    end = min(len(wav), idx[1] + pad)
    wav = wav[start:end]
    # 仅统计有声活动帧的 RMS，避免句中长停顿导致有声段被额外放大 +3~6 dB
    frame_rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256, center=True)[0]
    thr = max(float(np.percentile(frame_rms, 95)) * 0.15, 1e-3) if frame_rms.size > 0 else 1e-3
    active = frame_rms > thr
    speech_rms = float(np.sqrt(np.mean(frame_rms[active] ** 2) + 1e-8)) if np.any(active) else float(np.sqrt(np.mean(wav ** 2) + 1e-8))
    wav = wav * (0.11 / max(speech_rms, 1e-4))
    peak = float(np.max(np.abs(wav))) + 1e-8
    if peak > 0.92:
        wav = wav * (0.92 / peak)
    return wav.astype(np.float32)


def extract_f0_and_energy(wav: np.ndarray, sr: int = 24000, hop_length: int = 256, n_frames: int = None):
    """提取去倍频跳变的中值平滑 log_f0、log_energy 与精确清浊音标志 uv。"""
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

    # 3 点中值滤波 + 孤立单帧倍频/半频跳变抑制
    if len(f0) >= 3:
        f0 = scipy_signal.medfilt(f0, kernel_size=3).astype(np.float32)
        nb_mean = 0.5 * (f0[:-2] + f0[2:])
        ratio = f0[1:-1] / np.maximum(nb_mean, 1e-4)
        bad_oct = (ratio > 1.75) | (ratio < 0.57)
        if np.any(bad_oct):
            f0_mid = f0[1:-1].copy()
            f0_mid[bad_oct] = nb_mean[bad_oct]
            f0[1:-1] = f0_mid

    rms_gate = max(0.006, float(np.percentile(rms, 90)) * 0.06) if len(rms) > 0 else 0.006
    uv = (rms > rms_gate) & (f0 > 68.0) & (f0 < 580.0)
    # 消除孤立 1 帧伪浊音毛刺
    if len(uv) >= 3:
        isolated = uv[1:-1] & (~uv[:-2]) & (~uv[2:])
        uv[1:-1] = uv[1:-1] & (~isolated)

    log_f0 = np.zeros_like(f0, dtype=np.float32)
    if uv.any():
        voiced_idx = np.where(uv)[0]
        voiced_vals = np.log(np.clip(f0[voiced_idx], 65.0, 600.0))
        log_f0 = np.interp(np.arange(len(f0)), voiced_idx, voiced_vals).astype(np.float32)
    log_energy = np.log(np.clip(rms, 1e-5, None)).astype(np.float32)
    return log_f0, log_energy, uv.astype(np.float32)


# ------------------------------------------------------------------------------
# 完备中英双语与混读音素前端 (a5 / v3)
#   - 区分齐齿呼 i 与舌尖前/后元音 ii (zi/ci/si) / iii (zhi/chi/shi/ri)
#   - 还原 y/w 零声母与 iu->iou / ui->uei / un->uen / ue->ve 省写，覆盖全部 20,901 个 CJK 汉字零 <unk>
#   - 显式儿化韵尾 zh_er_coda + 完整「一 / 不 / A-一-A / 双音节韵律步三声」变调
#   - 用音素级词/韵律边界标记 boundary_ids (0: 音节内, 1: 词内音节首, 2: 词/韵律词/语种切换首) 替代抢占连读辅音帧的 <sp>
# ------------------------------------------------------------------------------
ZH_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
]
ZH_FINALS_BASE = [
    "a", "o", "e", "i", "ii", "iii", "u", "v", "er",
    "ai", "ei", "ao", "ou", "ia", "ie", "ua", "uo", "ve",
    "iao", "iou", "uai", "uei",
    "an", "en", "in", "vn", "ang", "eng", "ing", "ong",
    "ian", "uan", "van", "uen", "iang", "uang", "ueng", "iong",
]
EN_ARPABET = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", ",", ".", "?", "!", ";", ":"]

_Y_RESTORE = {
    "u": "v", "ue": "ve", "uan": "van", "un": "vn",
    "a": "ia", "e": "ie", "ao": "iao", "ou": "iou",
    "an": "ian", "ang": "iang", "ong": "iong",
}
_W_RESTORE = {
    "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
    "an": "uan", "en": "uen", "ang": "uang", "eng": "ueng",
}
_ABBREV_RESTORE = {"iu": "iou", "ui": "uei", "un": "uen"}
_NASAL_SYL_MAP = {"hm": ("h", "en"), "hng": ("h", "eng"), "m": ("m", "u"), "n": ("n", "en"), "ng": ("n", "eng")}

# 非儿化词汇表（防止把实义「儿 / 尔」误合并为前一字的儿化尾音）
NON_ERHUA_WORDS = {
    "儿童", "儿子", "儿女", "儿科", "儿歌", "儿媳", "儿孙", "儿戏", "儿时", "儿茶",
    "而且", "而后", "而是", "而今", "而已", "耳朵", "耳垂", "耳机", "耳光", "耳目",
    "女儿", "孤儿", "婴儿", "幼儿", "少儿", "新生儿", "胎儿", "产儿", "健儿", "宠儿",
    "男儿", "小儿", "育儿", "托儿", "哈尔滨", "首尔", "尼泊尔", "华尔街", "诺贝尔",
    "高尔夫", "马尔代夫", "阿尔法", "爱尔兰", "威尔士", "墨尔本", "荷尔蒙", "达尔文",
}


class BilingualPhonemeFrontend:
    def __init__(self):
        vocab = list(SPECIAL_TOKENS)
        vocab += [f"zh_{ini}" for ini in ZH_INITIALS]
        for fin in ZH_FINALS_BASE:
            for tone in ("1", "2", "3", "4", "5"):
                vocab.append(f"zh_{fin}{tone}")
        vocab.append("zh_er_coda")
        vocab += [f"en_{ph}" for ph in EN_ARPABET]
        self.vocab = vocab
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self.pad_id = self.tok2id["<pad>"]
        self.bos_id = self.tok2id["<bos>"]
        self.eos_id = self.tok2id["<eos>"]
        self.unk_id = self.tok2id["<unk>"]

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
        if n < 1000:
            rem = n % 100
            if rem == 0:
                return digits[n // 100] + "百"
            if rem < 10:
                return digits[n // 100] + "百零" + digits[rem]
            return digits[n // 100] + "百" + digits[rem // 10] + "十" + ("" if rem % 10 == 0 else digits[rem % 10])
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
    def is_erhua_pinyin(syl: str) -> bool:
        s = syl.strip().lower().replace("ü", "v").replace("u:", "v")
        body = s[:-1] if s and s[-1] in "12345" else s
        return len(body) > 1 and body.endswith("r") and body != "er"

    @staticmethod
    def _split_zh_syllable(syl: str):
        syl = syl.strip().lower().replace("ü", "v").replace("u:", "v").replace("ê", "e")
        if not syl:
            return []
        tone = syl[-1] if syl[-1] in "12345" else "5"
        body = syl[:-1] if syl[-1] in "12345" else syl
        if not body:
            return []
        if body in _NASAL_SYL_MAP:
            ini_n, fin_n = _NASAL_SYL_MAP[body]
            return [f"zh_{ini_n}", f"zh_{fin_n}{tone}"]

        has_erhua = False
        if len(body) > 1 and body.endswith("r") and body != "er":
            has_erhua = True
            body = body[:-1]

        if body == "er":
            return [f"zh_er{tone}"]

        ini = ""
        for cand in sorted(ZH_INITIALS, key=len, reverse=True):
            if body.startswith(cand) and len(body) > len(cand):
                ini = cand
                body = body[len(cand):]
                break

        if body == "i":
            if ini in ("z", "c", "s"):
                body = "ii"
            elif ini in ("zh", "ch", "sh", "r"):
                body = "iii"
        elif ini == "y" and body in _Y_RESTORE:
            body = _Y_RESTORE[body]
        elif ini == "w" and body in _W_RESTORE:
            body = _W_RESTORE[body]
        elif ini in ("j", "q", "x") and body in ("u", "ue", "uan", "un"):
            body = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}[body]
        elif ini in ("n", "l") and body == "ue":
            body = "ve"
        elif body in _ABBREV_RESTORE:
            body = _ABBREV_RESTORE[body]

        toks = []
        if ini:
            toks.append(f"zh_{ini}")
        toks.append(f"zh_{body}{tone}")
        if has_erhua:
            toks.append("zh_er_coda")
        return toks

    @staticmethod
    def _is_lexical_non_erhua(span: str, idx: int) -> bool:
        if idx <= 0 or span[idx] != "儿":
            return True
        for wlen in (3, 2):
            for st in range(max(0, idx - wlen + 1), min(len(span) - wlen + 1, idx + 1)):
                if span[st : st + wlen] in NON_ERHUA_WORDS:
                    return True
        return False

    @classmethod
    def _apply_zh_sandhi_and_erhua(cls, span: str, raw_pys: list):
        """在 (汉字串, pypinyin 音节列表) 上执行儿化合并 + 「一 / 不 / 叠字 / 双音节韵律步三声」变调。"""
        n = min(len(span), len(raw_pys))
        chars, pys = list(span[:n]), list(raw_pys[:n])

        # 1) 合并口语儿化尾音（如「这儿 / 一点儿 / 花儿 / 一会儿」-> zher4 / dianr3 / huar1 / huir4）
        m_chars, m_pys = [], []
        i = 0
        while i < len(chars):
            ch, py = chars[i], pys[i]
            if (
                i + 1 < len(chars)
                and chars[i + 1] == "儿"
                and pys[i + 1] in ("er2", "er5")
                and not py.startswith("er")
                and not cls._is_lexical_non_erhua(span, i + 1)
            ):
                tone = py[-1] if py and py[-1] in "12345" else "5"
                base = py[:-1] if py and py[-1] in "12345" else py
                m_chars.append(ch + "儿")
                m_pys.append(f"{base}r{tone}")
                i += 2
            else:
                m_chars.append(ch)
                m_pys.append(py)
                i += 1

        # 2) 「A 一 A」与「A 不 A」夹心轻声（如「看一看 / 好不好」）
        for k in range(1, len(m_chars) - 1):
            if m_chars[k] in ("一", "不") and m_chars[k - 1] == m_chars[k + 1]:
                m_pys[k] = m_pys[k][:-1] + "5"

        # 3) 「一 / 不」声调变调（对齐标贝真声发音习惯）
        digits_zh = set("零一二三四五六七八九十百千万亿")
        for k in range(len(m_chars)):
            ch = m_chars[k]
            if ch == "不" and m_pys[k].startswith("bu") and k + 1 < len(m_pys):
                nxt_t = m_pys[k + 1][-1] if m_pys[k + 1] else ""
                if nxt_t == "4" and m_pys[k][-1] != "5":
                    m_pys[k] = "bu2"
            elif ch == "一" and m_pys[k].startswith("yi") and m_pys[k][-1] != "5":
                prev_c = m_chars[k - 1][-1] if k > 0 else ""
                next_c = m_chars[k + 1][0] if k + 1 < len(m_chars) else ""
                if (
                    k + 1 >= len(m_pys)
                    or prev_c in ("第", "唯", "统", "初", "十", "零")
                    or next_c in ("月", "号", "楼", "班", "路", "年")
                    or (next_c in digits_zh and next_c not in ("一", "般", "起", "直", "定", "下", "样", "切", "致", "共", "点", "会"))
                ):
                    m_pys[k] = "yi1"
                else:
                    nxt_t = m_pys[k + 1][-1] if m_pys[k + 1] else ""
                    m_pys[k] = "yi2" if nxt_t == "4" else ("yi4" if nxt_t in ("1", "2", "3") else m_pys[k])

        # 4) 连续三声韵律步变调（双音节脚分组：2字->23，3字->223，4字->2323，避免长串机械全二声）
        k = 0
        while k < len(m_pys):
            if not m_pys[k].endswith("3"):
                k += 1
                continue
            r_end = k + 1
            while r_end < len(m_pys) and m_pys[r_end].endswith("3"):
                r_end += 1
            run_len = r_end - k
            if run_len == 2:
                m_pys[k] = m_pys[k][:-1] + "2"
            elif run_len == 3:
                m_pys[k] = m_pys[k][:-1] + "2"
                m_pys[k + 1] = m_pys[k + 1][:-1] + "2"
            elif run_len >= 4:
                pos = k
                while r_end - pos > 3:
                    m_pys[pos] = m_pys[pos][:-1] + "2"
                    pos += 2
                if r_end - pos == 3:
                    m_pys[pos] = m_pys[pos][:-1] + "2"
                    m_pys[pos + 1] = m_pys[pos + 1][:-1] + "2"
                elif r_end - pos == 2:
                    m_pys[pos] = m_pys[pos][:-1] + "2"
            k = r_end

        return m_pys

    def _zh_span_to_phonemes_and_bounds(self, span: str, first_bound: int = 2):
        raw_pys = [p[0] for p in pinyin(span, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore") if p]
        m_pys = self._apply_zh_sandhi_and_erhua(span, raw_pys)
        toks, bounds = [], []
        for idx, s in enumerate(m_pys):
            phs = self._split_zh_syllable(s)
            if not phs:
                continue
            # 短语首或每 2 个音节给韵律词起点 boundary=2，词内音节首给 boundary=1，韵母/儿化尾给 boundary=0
            b0 = first_bound if idx == 0 else (2 if idx % 2 == 0 else 1)
            for pi, ph in enumerate(phs):
                toks.append(ph)
                bounds.append(b0 if pi == 0 else 0)
        return toks, bounds

    def _en_word_to_phonemes_and_bounds(self, word: str, first_bound: int = 2):
        phs = []
        en_g2p = _get_en_g2p()
        if en_g2p is not None:
            for p in en_g2p(word):
                if f"en_{p}" in self.tok2id:
                    phs.append(f"en_{p}")
        if not phs:
            # 极简字母级回退（无 g2p_en 环境保底不崩）
            fallback_map = {"a": "AE1", "e": "EH1", "i": "IH1", "o": "AA1", "u": "AH1"}
            for ch in word.lower():
                ap = fallback_map.get(ch, ch.upper())
                if f"en_{ap}" in self.tok2id:
                    phs.append(f"en_{ap}")
        bounds = [(first_bound if i == 0 else 0) for i in range(len(phs))]
        return phs, bounds

    def parse_bznsyp_prosody(self, text_with_prosody: str, line_py: str):
        """逐字符儿化感知解析标贝标注：100% 保留句中逗号/分号停顿，并将 #1/#2/#3 韵律边界无损映射为 boundary_ids=2。"""
        norm_t = self.normalize_text(text_with_prosody)
        pys = line_py.strip().split()
        items = []  # list of (syl_or_punc, is_punc, bound)
        i, pi = 0, 0
        pending_bound = 2
        while i < len(norm_t):
            if norm_t[i] == "#" and i + 1 < len(norm_t) and norm_t[i + 1] in "1234":
                if norm_t[i + 1] in "123":
                    pending_bound = 2
                i += 2
                continue
            ch = norm_t[i]
            if ch in (",", ".", "?", "!", ";", ":"):
                items.append((ch, True, 0))
                pending_bound = 2
                i += 1
                continue
            if re.match(r"[\u4e00-\u9fff]", ch) and pi < len(pys):
                py = pys[pi]
                pi += 1
                i += 1
                # 若当前拼音带儿化 r，跳过紧随其后的「儿」字（中间允许隔 #1~#4 韵律符）
                if self.is_erhua_pinyin(py):
                    j = i
                    while j + 1 < len(norm_t) and norm_t[j] == "#" and norm_t[j + 1] in "1234":
                        j += 2
                    if j < len(norm_t) and norm_t[j] == "儿":
                        i = j + 1
                items.append((py, False, pending_bound))
                pending_bound = 1
                continue
            i += 1
        while pi < len(pys):
            items.append((pys[pi], False, pending_bound))
            pending_bound = 1
            pi += 1
        return items

    def text_to_ids(self, text: str, prosody_pinyin: list = None, bznsyp_prosody_items: list = None):
        """返回 (phoneme_ids, lang_ids, boundary_ids) 三元组（等长列表）。"""
        tokens = ["<bos>"]
        langs = [0]
        bounds = [0]

        if bznsyp_prosody_items is not None:
            for token_str, is_punc, b_flag in bznsyp_prosody_items:
                if is_punc:
                    if token_str in self.tok2id:
                        tokens.append(token_str); langs.append(0); bounds.append(0)
                else:
                    phs = self._split_zh_syllable(token_str)
                    for idx_p, ph in enumerate(phs):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(1)
                        bounds.append(b_flag if idx_p == 0 else 0)
        elif prosody_pinyin is not None:
            pending_b = 2
            for item in prosody_pinyin:
                if item in ("#1", "#2", "#3", "#4"):
                    pending_b = 2
                elif item in self.tok2id:
                    tokens.append(item); langs.append(0); bounds.append(0)
                    pending_b = 2
                else:
                    phs = self._split_zh_syllable(item)
                    for idx_p, ph in enumerate(phs):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(1)
                        bounds.append(pending_b if idx_p == 0 else 0)
                    pending_b = 1
        else:
            norm = self.normalize_text(text)
            pattern = re.compile(r"([\u4e00-\u9fff]+|[A-Za-z][A-Za-z'\-]*|#[1-4]|[,\.\?!;:])")
            pending_b = 2
            prev_lang = 0
            for m in pattern.finditer(norm):
                seg = m.group(0)
                if seg in ("#1", "#2", "#3", "#4"):
                    pending_b = 2
                elif re.match(r"^[\u4e00-\u9fff]+$", seg):
                    fb = 2 if (pending_b == 2 or prev_lang != 1) else 1
                    phs, bds = self._zh_span_to_phonemes_and_bounds(seg, first_bound=fb)
                    for ph, bd in zip(phs, bds):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(1)
                        bounds.append(bd)
                    prev_lang = 1
                    pending_b = 1
                elif re.match(r"^[A-Za-z]", seg):
                    phs, bds = self._en_word_to_phonemes_and_bounds(seg, first_bound=2)
                    for ph, bd in zip(phs, bds):
                        tokens.append(ph if ph in self.tok2id else "<unk>")
                        langs.append(2)
                        bounds.append(bd)
                    prev_lang = 2
                    pending_b = 2
                else:
                    if seg in self.tok2id:
                        tokens.append(seg); langs.append(0); bounds.append(0)
                    prev_lang = 0
                    pending_b = 2

        tokens.append("<eos>")
        langs.append(0)
        bounds.append(0)
        return [self.tok2id.get(t, self.unk_id) for t in tokens], langs, bounds


FRONTEND = BilingualPhonemeFrontend()
if __name__ == "__main__":  # FRONTEND_DEMO_PRINT
    print(f"Unified Bilingual Vocab Size (a5): {FRONTEND.vocab_size}")
    for demo in ["你好，请问今天北京的天气怎么样？", "Hello world, how is your API latency today?", "帮我把 Wi-Fi 切换到 5G，一会儿见！"]:
        pids, lids, bids = FRONTEND.text_to_ids(demo)
        print(f"  {demo} -> {len(pids)} phonemes (unks={sum(1 for x in pids if x == FRONTEND.unk_id)})")

# #############################################################################
# Cell 5: 核心模型（ChunkLookaheadBackbone 主力 + 保留主干 + 浊音感知声学模型 + 谐波源因果声码器）
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


class LookaheadConv1d(nn.Module):
    """k_left 步左上下文 + lookahead 步有限右前瞻的 Conv1d（集中式前瞻：只允许这一层看右边）。"""

    def __init__(self, in_ch: int, out_ch: int, k_left: int, lookahead: int, groups: int = 1):
        super().__init__()
        self.k_left = k_left
        self.lookahead = lookahead
        self.kernel_size = k_left + 1 + lookahead
        self.in_ch = in_ch
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.k_left, self.lookahead)))

    def init_state(self, batch_size: int, device, dtype):
        return torch.zeros(batch_size, self.in_ch, self.k_left, device=device, dtype=dtype)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor, is_final: bool = False):
        buf = torch.cat([state, x], dim=-1) if x.shape[-1] > 0 else state
        if is_final and self.lookahead > 0:
            buf = F.pad(buf, (0, self.lookahead))
        n_emit = buf.shape[-1] - self.kernel_size + 1
        if n_emit <= 0:
            return torch.zeros(buf.shape[0], self.conv.out_channels, 0, device=buf.device, dtype=buf.dtype), buf
        keep_len = self.k_left + self.lookahead
        new_state = buf[:, :, -keep_len:] if keep_len > 0 else buf[:, :, :0]
        return self.conv(buf), new_state


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
# 5.1b [归档保留] 门控 DeltaNet（E4）训练期 log 空间分块并行
# ==============================================================================
SCAN_CHUNK = 32


def pad_to_chunk(x: torch.Tensor, chunk: int = SCAN_CHUNK, value: float = 0.0) -> torch.Tensor:
    pad = (-x.shape[1]) % chunk
    return F.pad(x, (0, 0) * (x.dim() - 2) + (0, pad), value=value) if pad else x


def unit_lower_tri_inverse(N: torch.Tensor) -> torch.Tensor:
    L = N.shape[-1]
    P, A = N, torch.eye(L, device=N.device, dtype=N.dtype).expand_as(N)
    for _ in range(max(1, math.ceil(math.log2(L)))):
        A = A + P @ A
        P = P @ P
    return A


def gated_delta_rule_chunkwise(q, k, v, beta, log_alpha, S_init, chunk: int = SCAN_CHUNK):
    T = q.shape[1]
    q, k, v, beta, log_alpha = (pad_to_chunk(t, chunk) for t in (q, k, v, beta, log_alpha))
    S, ys = S_init, []
    for s0 in range(0, q.shape[1], chunk):
        qc, kc, vc = (t[:, s0:s0 + chunk].transpose(1, 2) for t in (q, k, v))
        bc, la = beta[:, s0:s0 + chunk].transpose(1, 2), log_alpha[:, s0:s0 + chunk].transpose(1, 2)
        L = qc.shape[2]
        tri = torch.ones(L, L, device=q.device, dtype=torch.bool)
        strict_mask = torch.tril(tri, diagonal=-1)
        causal_mask = torch.tril(tri)
        log_G = torch.cumsum(la, dim=-1)
        G = log_G.exp()
        log_ratio = log_G[..., :, None] - log_G[..., None, :]
        ratio_strict = torch.where(strict_mask, log_ratio, torch.zeros_like(log_ratio)).exp() * strict_mask
        ratio_causal = torch.where(causal_mask, log_ratio, torch.zeros_like(log_ratio)).exp() * causal_mask
        tail_ratio = (log_G[..., -1:] - log_G).exp()
        T_mat = bc[..., None] * torch.einsum("bhik,bhjk->bhij", kc, kc) * ratio_strict
        R = bc[..., None] * (vc - G[..., None] * torch.einsum("bhik,bhkv->bhiv", kc, S))
        W = unit_lower_tri_inverse(-T_mat) @ R
        y = G[..., None] * torch.einsum("bhik,bhkv->bhiv", qc, S) \
            + (torch.einsum("bhik,bhjk->bhij", qc, kc) * ratio_causal) @ W
        S = G[..., -1][..., None, None] * S + torch.einsum("bhik,bhi,bhiv->bhkv", kc, tail_ratio, W)
        ys.append(y)
    return torch.cat(ys, dim=2).transpose(1, 2)[:, :T], S


def gated_delta_rule_recurrent(q, k, v, beta, log_alpha, S):
    ys = []
    for t in range(q.shape[1]):
        a, b = torch.exp(log_alpha[:, t])[..., None, None], beta[:, t][..., None, None]
        S_dec = a * S
        S = S_dec + b * torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S_dec))
        ys.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(ys, dim=1), S


# ==============================================================================
# 5.2 序列主干：主力 E3 ChunkLookaheadBackbone（局部窗口相对 RoPE，永久 O(1) 状态）+ 归档保留主干
# ==============================================================================
def apply_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    dh = x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    freqs = torch.outer(pos.to(torch.float32), inv_freq)
    cos = freqs.cos()[None, None, :, :].to(x.dtype)
    sin = freqs.sin()[None, None, :, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def _apply_rope_with_cache(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class OfflineBiTransformerBackbone(nn.Module):
    """[归档保留] E0 非流式全双向 Transformer"""

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
    """[归档保留] E1 Causal ConvNeXt + Uni-GRU"""

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
    """[归档保留] E2 滑窗自注意力（同步升级为未旋转 K 缓存 + 局部窗口相对 RoPE，无位置溢出）"""

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
        scores = scores.masked_fill(~allow[None, None, :, :], torch.finfo(scores.dtype).min)
        out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, T, D)
        return self.conv_ffn(x + self.proj(out))

    def init_state(self, batch_size: int, device, dtype):
        return {
            "k": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "conv": self.conv_ffn.init_state(batch_size, device, dtype),
        }

    def forward_step(self, x: torch.Tensor, state: dict):
        B, C, D = x.shape
        if C == 0:
            return x, state
        qkv = self.qkv(self.norm1(x)).view(B, C, 3, self.n_heads, self.head_dim)
        q_raw, k_raw, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        k_cat_raw = torch.cat([state["k"], k_raw], dim=2)
        v_all = torch.cat([state["v"], v], dim=2)
        k_len = k_cat_raw.shape[2]
        pos_k = torch.arange(k_len, device=x.device)
        pos_q = torch.arange(k_len - C, k_len, device=x.device)
        q = apply_rope(q_raw, pos_q)
        k_all = apply_rope(k_cat_raw, pos_k)
        diff = pos_q[:, None] - pos_k[None, :]
        allow = (diff >= 0) & (diff < self.window_size)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allow[None, None, :, :], torch.finfo(scores.dtype).min)
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).contiguous().view(B, C, D)
        x, new_conv = self.conv_ffn.forward_step(x + self.proj(out), state["conv"])
        keep = self.window_size - 1
        return x, {
            "k": k_cat_raw[:, :, -keep:, :] if keep > 0 else k_cat_raw[:, :, :0, :],
            "v": v_all[:, :, -keep:, :] if keep > 0 else v_all[:, :, :0, :],
            "conv": new_conv,
        }


class SWABackbone(nn.Module):
    """[归档保留] E2 浅前瞻 + SWA"""

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


class ChunkLookaheadLayer(nn.Module):
    """★ 主力单层分块深前瞻注意力 + 因果 ConvNeXt（a5 局部窗口相对 RoPE 升级版）：
    - 在当前 [commit_chunk C + lookahead L] 窗口内做完整双向自注意力，同时因果读取长度 <= W 的历史 KV Cache；
    - KV Cache 存储【未旋转 K】，每次将当前窗口锚定在局部坐标 W、历史锚定在 [W-past_len, W)，固定 RoPE 表长仅需 <= 128，
      多句连续流式数小时不重置状态亦严格 O(1)、零表重分配、零大角度外推误差；
    - 算完仅提交前 commit_len 个 token 的 KV 与卷积缓存，彻底丢弃 lookahead 状态，多层堆叠零前瞻膨胀；
    - 训练期 forward_windows 层主序零 chunk 循环全并行，与流式 forward_chunk 逐元素数学等价。"""

    def __init__(self, d_model: int = 256, n_heads: int = 4, window_size: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv = CausalConvNeXtBlock(d_model, kernel_size=5, mult=2)
        _pos = torch.arange(128, dtype=torch.float32)
        _inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        _freqs = torch.outer(_pos, _inv)
        self.register_buffer("_rope_cos", _freqs.cos(), persistent=False)
        self.register_buffer("_rope_sin", _freqs.sin(), persistent=False)

    def _get_local_rope(self, needed_len: int, device: torch.device, dtype: torch.dtype):
        if self._rope_cos.shape[0] < needed_len or self._rope_cos.device != device:
            cap = max(needed_len, 128)
            with torch.inference_mode(False), torch.no_grad():
                pos = torch.arange(cap, device=device, dtype=torch.float32)
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
                freqs = torch.outer(pos, inv_freq)
                self._rope_cos = freqs.cos()
                self._rope_sin = freqs.sin()
        return self._rope_cos[:needed_len].to(dtype), self._rope_sin[:needed_len].to(dtype)

    def init_state(self, batch_size: int, device, dtype):
        return {
            "k": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=device, dtype=dtype),
            "conv": self.conv.init_state(batch_size, device, dtype),
        }

    def forward_chunk(self, x_win: torch.Tensor, state: dict, commit_len: int, win_mask: torch.Tensor = None, output_commit_only: bool = False):
        B, M, D = x_win.shape
        W = self.window_size
        xn = self.norm1(x_win)
        if output_commit_only and commit_len < M and type(self.qkv) is nn.Linear:
            w_q, w_kv = self.qkv.weight.split([D, 2 * D], dim=0)
            q = F.linear(xn[:, :commit_len, :], w_q).reshape(B, commit_len, self.n_heads, self.head_dim).transpose(1, 2)
            kv = F.linear(xn, w_kv).reshape(B, M, 2, self.n_heads, self.head_dim)
            k_raw, v = kv.permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(xn).reshape(B, M, 3, self.n_heads, self.head_dim)
            q, k_raw, v = qkv.permute(2, 0, 3, 1, 4)
            if output_commit_only and commit_len < M:
                q = q[:, :, :commit_len, :]
        q_len = q.shape[2]
        past_len = state["k"].shape[2]
        k_cat_raw = torch.cat([state["k"], k_raw], dim=2)
        v_all = torch.cat([state["v"], v], dim=2)

        cos, sin = self._get_local_rope(W + M, x_win.device, q.dtype)
        off = W - past_len
        q_rot = _apply_rope_with_cache(q, cos[W : W + q_len][None, None, :, :], sin[W : W + q_len][None, None, :, :])
        k_all = _apply_rope_with_cache(k_cat_raw, cos[off : off + past_len + M][None, None, :, :], sin[off : off + past_len + M][None, None, :, :])

        scores = torch.matmul(q_rot, k_all.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if win_mask is not None:
            key_valid = torch.cat([torch.ones(B, past_len, device=x_win.device, dtype=torch.bool), win_mask], dim=1)
            scores = scores.masked_fill(~key_valid[:, None, None, :], torch.finfo(scores.dtype).min)
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).reshape(B, q_len, D)
        h = x_win[:, :q_len, :] + self.out_proj(out)

        pad_left = self.conv.dwconv.pad_left
        buf = torch.cat([state["conv"], h.transpose(1, 2)], dim=-1)
        conv_out = self.conv.dwconv.conv(buf).transpose(1, 2)
        h_out = h + self.conv.pw2(F.gelu(self.conv.pw1(self.conv.norm(conv_out))))
        new_conv = buf[:, :, commit_len : commit_len + pad_left] if pad_left > 0 else state["conv"]

        new_k = k_cat_raw[:, :, :past_len + commit_len, :][:, :, -W:, :] if W > 0 else k_cat_raw[:, :, :0, :]
        new_v = v_all[:, :, :past_len + commit_len, :][:, :, -W:, :] if W > 0 else v_all[:, :, :0, :]
        return h_out, {"k": new_k, "v": new_v, "conv": new_conv}

    def forward_windows(
        self,
        x_win: torch.Tensor,
        win_mask: torch.Tensor,
        past_valid: torch.Tensor,
        chunk_size: int,
        output_commit_only: bool = False,
    ) -> torch.Tensor:
        """层主序全窗口零循环并行前向（局部窗口相对 RoPE + Attention + ConvNeXt 同步向量化）。"""
        B, N, M, D = x_win.shape
        C, W = chunk_size, self.window_size
        H, dh = self.n_heads, self.head_dim
        q_len = C if (output_commit_only and C < M) else M

        xn = self.norm1(x_win)
        if output_commit_only and C < M and type(self.qkv) is nn.Linear:
            w_q, w_kv = self.qkv.weight.split([D, 2 * D], dim=0)
            q = F.linear(xn[:, :, :C, :], w_q).reshape(B, N, C, H, dh).permute(0, 1, 3, 2, 4)
            kv = F.linear(xn, w_kv).reshape(B, N, M, 2, H, dh)
            k_raw = kv[:, :, :, 0].permute(0, 1, 3, 2, 4)
            v = kv[:, :, :, 1].permute(0, 1, 3, 2, 4)
        else:
            qkv = self.qkv(xn).reshape(B, N, M, 3, H, dh)
            q = qkv[:, :, :q_len, 0].permute(0, 1, 3, 2, 4)
            k_raw = qkv[:, :, :, 1].permute(0, 1, 3, 2, 4)
            v = qkv[:, :, :, 2].permute(0, 1, 3, 2, 4)

        if W > 0:
            k_commit = k_raw[:, :, :, :C, :].permute(0, 2, 1, 3, 4).reshape(B, H, N * C, dh)
            v_commit = v[:, :, :, :C, :].permute(0, 2, 1, 3, 4).reshape(B, H, N * C, dh)
            end_p = W + (N - 1) * C
            k_past = F.pad(k_commit, (0, 0, W, 0))[:, :, :end_p, :].unfold(2, W, C).permute(0, 2, 1, 4, 3)
            v_past = F.pad(v_commit, (0, 0, W, 0))[:, :, :end_p, :].unfold(2, W, C).permute(0, 2, 1, 4, 3)
            k_cat_raw = torch.cat([k_past, k_raw], dim=3)
            v_all = torch.cat([v_past, v], dim=3)
            key_valid = torch.cat([past_valid[None, :, :].expand(B, -1, -1), win_mask], dim=2)
        else:
            k_cat_raw, v_all, key_valid = k_raw, v, win_mask

        cos, sin = self._get_local_rope(W + M, x_win.device, q.dtype)
        q_rot = _apply_rope_with_cache(q, cos[W : W + q_len][None, None, None, :, :], sin[W : W + q_len][None, None, None, :, :])
        k_all = _apply_rope_with_cache(k_cat_raw, cos[: W + M][None, None, None, :, :], sin[: W + M][None, None, None, :, :])

        out = F.scaled_dot_product_attention(
            q_rot.reshape(B * N, H, q_len, dh),
            k_all.reshape(B * N, H, W + M, dh),
            v_all.reshape(B * N, H, W + M, dh),
            attn_mask=key_valid.reshape(B * N, 1, 1, W + M),
        )
        out = out.transpose(1, 2).reshape(B, N, q_len, D)
        h = x_win[:, :, :q_len, :] + self.out_proj(out)

        pad_left = self.conv.dwconv.pad_left
        if pad_left > 0:
            h_commit = h[:, :, :C, :].reshape(B, N * C, D)
            end_c = pad_left + (N - 1) * C
            conv_state = F.pad(h_commit, (0, 0, pad_left, 0))[:, :end_c, :].unfold(1, pad_left, C)
            buf = torch.cat([conv_state, h.transpose(2, 3)], dim=3).reshape(B * N, D, pad_left + q_len)
        else:
            buf = h.transpose(2, 3).reshape(B * N, D, q_len)
        conv_out = self.conv.dwconv.conv(buf).transpose(1, 2).reshape(B, N, q_len, D)
        return h + self.conv.pw2(F.gelu(self.conv.pw1(self.conv.norm(conv_out))))


class ChunkLookaheadBackbone(nn.Module):
    """★ AetherTTS 唯一主力流式输入主干：5 层深双向分块前瞻注意力（C=4, L=8, W=48）。"""

    def __init__(self, d_model: int = 256, chunk_size: int = 4, lookahead: int = 8, n_layers: int = 5, n_heads: int = 4, window_size: int = 48):
        super().__init__()
        self.is_streaming = True
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.lookahead = lookahead
        self.layers = nn.ModuleList([ChunkLookaheadLayer(d_model, n_heads=n_heads, window_size=window_size) for _ in range(n_layers)])

    def _run_window(self, win: torch.Tensor, layer_states: list, commit_len: int, win_mask: torch.Tensor = None):
        h = win
        new_layers = []
        last_idx = len(self.layers) - 1
        for idx, (l, st) in enumerate(zip(self.layers, layer_states)):
            h, nst = l.forward_chunk(h, st, commit_len=commit_len, win_mask=win_mask, output_commit_only=(idx == last_idx))
            new_layers.append(nst)
        return h[:, :commit_len, :], new_layers

    def _forward_sequential(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        layer_states = [l.init_state(B, x.device, x.dtype) for l in self.layers]
        emits = []
        for s in range(0, T, self.chunk_size):
            c_len = min(self.chunk_size, T - s)
            e = min(T, s + self.chunk_size + self.lookahead)
            win_mask = mask[:, s:e] if mask is not None else None
            h_commit, layer_states = self._run_window(x[:, s:e, :], layer_states, commit_len=c_len, win_mask=win_mask)
            emits.append(h_commit)
        return torch.cat(emits, dim=1) if emits else x[:, :0, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        if T == 0:
            return x[:, :0, :]
        C, L = self.chunk_size, self.lookahead
        M = C + L
        N = (T + C - 1) // C
        T_pad = (N - 1) * C + M
        pad_r = T_pad - T
        x_pad = F.pad(x, (0, 0, 0, pad_r)) if pad_r > 0 else x
        if mask is None:
            mask_pad = (torch.arange(T_pad, device=x.device)[None, :] < T).expand(B, -1)
        else:
            mask_pad = F.pad(mask, (0, pad_r), value=False) if pad_r > 0 else mask

        x_win = x_pad.unfold(1, M, C).transpose(2, 3).contiguous()
        win_mask = mask_pad.unfold(1, M, C).contiguous()

        W = self.layers[0].window_size
        if W > 0:
            chunk_starts = torch.arange(N, device=x.device)[:, None] * C
            past_valid = (chunk_starts - W + torch.arange(W, device=x.device)[None, :]) >= 0
        else:
            past_valid = torch.zeros(N, 0, device=x.device, dtype=torch.bool)

        h = x_win
        last_idx = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            h = layer.forward_windows(
                h, win_mask, past_valid,
                chunk_size=C, output_commit_only=(idx == last_idx),
            )
        return h[:, :, :C, :].reshape(B, N * C, D)[:, :T, :]

    def init_state(self, batch_size: int, device, dtype):
        return {
            "buf": torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype),
            "layers": [l.init_state(batch_size, device, dtype) for l in self.layers],
        }

    def forward_committed_with_lookahead(self, x_commit: torch.Tensor, x_lookahead: torch.Tensor, state: dict):
        """Phase 2 流式前端原生接口：显式区分已确认提交音素与暂定前瞻音素（前瞻永不写入缓存，下一拍可被前端无损修正）。"""
        c_len = x_commit.shape[1]
        if c_len == 0:
            return x_commit, state
        win = torch.cat([x_commit, x_lookahead], dim=1) if x_lookahead.shape[1] > 0 else x_commit
        h_commit, new_layers = self._run_window(win, state["layers"], commit_len=c_len)
        return h_commit, {"buf": state["buf"], "layers": new_layers}

    def forward_step(self, x_new: torch.Tensor, state: dict, is_final: bool = False):
        buf = torch.cat([state["buf"], x_new], dim=1) if x_new.shape[1] > 0 else state["buf"]
        layer_states = state["layers"]
        emits = []
        while buf.shape[1] >= self.chunk_size + self.lookahead:
            win = buf[:, :self.chunk_size + self.lookahead, :]
            h_commit, layer_states = self._run_window(win, layer_states, commit_len=self.chunk_size)
            emits.append(h_commit)
            buf = buf[:, self.chunk_size:, :]
        if is_final:
            while buf.shape[1] > 0:
                c_len = min(self.chunk_size, buf.shape[1])
                win = buf[:, :c_len + self.lookahead, :]
                h_commit, layer_states = self._run_window(win, layer_states, commit_len=c_len)
                emits.append(h_commit)
                buf = buf[:, c_len:, :]
        out = torch.cat(emits, dim=1) if emits else torch.zeros(buf.shape[0], 0, self.d_model, device=buf.device, dtype=buf.dtype)
        return out, {"buf": buf, "layers": layer_states}


class GatedDeltaNetBlock(nn.Module):
    """[归档保留] 标准门控 DeltaNet"""

    def __init__(self, d_model: int = 256, n_heads: int = 4, ffn_hidden: int = 384, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.qkv_conv = CausalConv1d(d_model * 3, d_model * 3, kernel_size=d_conv, groups=d_model * 3)
        self.ba_proj = nn.Linear(d_model, n_heads * 2, bias=True)
        with torch.no_grad():
            self.ba_proj.bias[:n_heads].zero_()
            self.ba_proj.bias[n_heads:].fill_(2.0)
        self.z = nn.Linear(d_model, d_model, bias=False)
        self.o_norm = nn.GroupNorm(n_heads, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_down = nn.Linear(ffn_hidden, d_model, bias=False)

    def _run(self, x: torch.Tensor, attn, conv_state, S_init):
        B, T, D = x.shape
        xn = self.norm1(x)
        qkv_in = self.qkv(xn).transpose(1, 2)
        if conv_state is None:
            qkv_c, new_conv = F.silu(self.qkv_conv(qkv_in).transpose(1, 2)), None
        else:
            qkv_raw, new_conv = self.qkv_conv.forward_step(qkv_in, conv_state)
            qkv_c = F.silu(qkv_raw.transpose(1, 2))
        q, k, v = qkv_c.reshape(B, T, 3, self.n_heads, self.head_dim).float().unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        b_logit, a_logit = self.ba_proj(xn).float().reshape(B, T, 2, self.n_heads).unbind(2)
        beta, log_alpha = torch.sigmoid(b_logit), F.logsigmoid(a_logit)
        if S_init is None:
            S_init = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        o, S = attn(q, k, v, beta, log_alpha, S_init)
        y = self.o_norm(o.reshape(B * T, D).to(x.dtype)).reshape(B, T, D) * F.silu(self.z(xn))
        h = x + self.out_proj(y)
        xc = self.norm2(h)
        return h + self.ffn_down(F.silu(self.ffn_gate(xc)) * self.ffn_up(xc)), new_conv, S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, gated_delta_rule_chunkwise, None, None)[0]

    def init_state(self, batch_size: int, device, dtype):
        return {
            "conv": self.qkv_conv.init_state(batch_size, device, dtype),
            "S": torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32),
        }

    def forward_step(self, x: torch.Tensor, state: dict):
        if x.shape[1] == 0:
            return x, state
        out, new_conv, S = self._run(x, gated_delta_rule_recurrent, state["conv"], state["S"])
        return out, {"conv": new_conv, "S": S}


class GatedDeltaNetBackbone(nn.Module):
    """[归档保留] E4 浅前瞻 + 标准门控 DeltaNet"""

    def __init__(self, d_model: int = 256, k_left: int = 3, lookahead: int = 4, n_layers: int = 5, n_heads: int = 4):
        super().__init__()
        self.is_streaming = True
        self.lookahead_layer = CentralizedLookaheadBlock(d_model, k_left=k_left, lookahead=lookahead, mult=2)
        self.layers = nn.ModuleList([GatedDeltaNetBlock(d_model, n_heads=n_heads, ffn_hidden=384, d_conv=4) for _ in range(n_layers)])

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
# 5.3 单调对齐器（掩码防填充泄漏 + 位置消歧）& 浊音感知声学模型 AetherAcousticModel (a5)
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


@torch.no_grad()
def build_beta_binomial_log_prior(text_lens: torch.Tensor, mel_lens: torch.Tensor, max_t: int, max_f: int, scaling: float = 1.0) -> torch.Tensor:
    B = text_lens.shape[0]
    device = text_lens.device
    f_col = torch.arange(max_f, device=device, dtype=torch.float32).view(1, max_f, 1)
    t_row = torch.arange(max_t, device=device, dtype=torch.float32).view(1, 1, max_t)
    n = (text_lens.to(torch.float32) - 1.0).view(B, 1, 1)
    f_b = mel_lens.to(torch.float32).view(B, 1, 1)
    alpha = scaling * (f_col + 1.0)
    beta = scaling * (f_b - f_col)
    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(t_row + 1.0) - torch.lgamma(n - t_row + 1.0)
    log_beta_num = torch.lgamma(t_row + alpha) + torch.lgamma(n - t_row + beta) - torch.lgamma(n + alpha + beta)
    log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    dense = (log_comb + log_beta_num - log_beta_den).clamp_(min=-50.0)
    valid = (f_col < f_b) & (t_row < text_lens.to(torch.float32).view(B, 1, 1))
    single_tok = (text_lens <= 1).view(B, 1, 1)
    out = torch.full((B, max_f, max_t), -1e4, device=device, dtype=torch.float32)
    out = torch.where(valid & ~single_tok, dense, out)
    out = torch.where(valid & single_tok, torch.zeros_like(out), out)
    return out


class AlignmentEncoder(nn.Module):
    """带有效帧掩码与轻量正弦位置编码的单调对齐编码器（消除批右填充污染与叠词同音素混淆）。"""

    def __init__(self, n_mels: int = 100, d_text: int = 256, d_attn: int = 80, scale: float = 10.0):
        super().__init__()
        self.d_text = d_text
        self.key_proj = nn.Sequential(nn.Conv1d(d_text, d_text, 3, padding=1), nn.ReLU(), nn.Conv1d(d_text, d_attn, 1))
        self.query_proj = nn.Sequential(nn.Conv1d(n_mels, n_mels * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(n_mels * 2, d_attn, 1))
        self.scale = scale

    @staticmethod
    def _sinusoidal_pos(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(1000.0) / dim))
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype)

    def forward(self, mel: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor, attn_log_prior: torch.Tensor = None, mel_mask: torch.Tensor = None):
        T_max = text_emb.shape[1]
        pos_pe = self._sinusoidal_pos(T_max, self.d_text, text_emb.device, text_emb.dtype)[None, :, :]
        t_in = (text_emb + 0.15 * pos_pe) * text_mask[:, :, None].to(text_emb.dtype)
        m_in = mel * mel_mask[:, None, :].to(mel.dtype) if mel_mask is not None else mel
        keys = F.normalize(self.key_proj(t_in.transpose(1, 2)).float(), dim=1)
        queries = F.normalize(self.query_proj(m_in).float(), dim=1)
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


@torch.no_grad()
def viterbi_monotonic_alignment(attn_logprob: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor):
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


FRAME_PHASE_DIM = 24


def frame_phase_and_dur(durations: torch.Tensor, max_frames: int):
    B, T = durations.shape
    device = durations.device
    if T == 0 or max_frames == 0:
        return (
            torch.zeros(B, max_frames, device=device, dtype=torch.long),
            torch.ones(B, max_frames, device=device, dtype=torch.long),
        )
    d = durations.long().clamp(min=0)
    ends = torch.cumsum(d, dim=1)
    starts = ends - d
    idx = torch.arange(max_frames, device=device, dtype=torch.long).unsqueeze(0).expand(B, max_frames).contiguous()
    tok_idx = torch.searchsorted(ends, idx, right=True)
    valid = tok_idx < T
    tok_safe = tok_idx.clamp(max=T - 1)
    s_f = starts.gather(1, tok_safe)
    d_f = d.gather(1, tok_safe)
    tau = torch.where(valid, idx - s_f, torch.zeros_like(idx))
    dur_f = torch.where(valid, d_f.clamp(min=1), torch.ones_like(idx))
    return tau, dur_f


def build_frame_phase_features(durations: torch.Tensor, max_frames: int) -> torch.Tensor:
    tau, dur_f = frame_phase_and_dur(durations, max_frames)
    tau_f = tau.to(torch.float32)
    dur_ff = dur_f.to(torch.float32)
    rem_f = (dur_ff - 1.0 - tau_f).clamp(min=0.0)
    prog = (tau_f + 0.5) / dur_ff

    prog_angles = prog.unsqueeze(-1) * torch.tensor([2.0 * math.pi * k for k in (1, 2, 4, 8)], device=durations.device, dtype=torch.float32)
    inv_scales = torch.tensor([1.0 / 2.0, 1.0 / 6.0, 1.0 / 18.0], device=durations.device, dtype=torch.float32)
    tau_angles = tau_f.unsqueeze(-1) * inv_scales
    rem_angles = rem_f.unsqueeze(-1) * inv_scales
    scalars = torch.stack([
        (tau_f / 32.0).clamp(max=2.0),
        (rem_f / 32.0).clamp(max=2.0),
        prog,
        torch.log1p(dur_ff) / 4.0,
    ], dim=-1)
    return torch.cat([
        torch.sin(prog_angles), torch.cos(prog_angles),
        torch.sin(tau_angles), torch.cos(tau_angles),
        torch.sin(rem_angles), torch.cos(rem_angles),
        scalars,
    ], dim=-1)


class CausalPredictorHead(nn.Module):
    def __init__(self, d_model: int = 256, kernel_size: int = 3, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.c1 = CausalConv1d(d_model, d_model, kernel_size)
        self.n1 = nn.LayerNorm(d_model)
        self.c2 = CausalConv1d(d_model, d_model, kernel_size)
        self.n2 = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.c1(x.transpose(1, 2)).transpose(1, 2)))
        h = F.silu(self.n2(self.c2(h.transpose(1, 2)).transpose(1, 2)))
        out = self.proj(h)
        return out.squeeze(-1) if self.out_dim == 1 else out

    def init_state(self, batch_size: int, device, dtype):
        return (self.c1.init_state(batch_size, device, dtype), self.c2.init_state(batch_size, device, dtype))

    def forward_step(self, x: torch.Tensor, state):
        if x.shape[1] == 0:
            shape = (x.shape[0], 0) if self.out_dim == 1 else (x.shape[0], 0, self.out_dim)
            return torch.zeros(*shape, device=x.device, dtype=x.dtype), state
        h, s1 = self.c1.forward_step(x.transpose(1, 2), state[0])
        h = F.silu(self.n1(h.transpose(1, 2)))
        h, s2 = self.c2.forward_step(h.transpose(1, 2), state[1])
        h = F.silu(self.n2(h.transpose(1, 2)))
        out = self.proj(h)
        return (out.squeeze(-1) if self.out_dim == 1 else out), (s1, s2)


def mel_structure_loss(mel_pred: torch.Tensor, mel_gt: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
    """2D 频谱共振峰与时间瞬态结构损失：抑制纯逐点 L1 的过平滑模糊，锐化元音共振峰脊线与塞音/擦音边界。"""
    n_mels = mel_pred.shape[1]
    # 1) 频率轴一阶差分 L1（约束相邻梅尔频带斜率，拉起共振峰峰谷对比度）
    df_p = mel_pred[:, 1:, :] - mel_pred[:, :-1, :]
    df_g = mel_gt[:, 1:, :] - mel_gt[:, :-1, :]
    loss_freq = (F.l1_loss(df_p, df_g, reduction="none") * mel_mask[:, None, :]).sum() / (mel_mask.sum().clamp(min=1) * max(n_mels - 1, 1))
    # 2) 时间轴一阶差分 L1（约束辅音爆发与协同发音瞬态斜率）
    if mel_pred.shape[-1] > 1:
        t_pair_mask = mel_mask[:, 1:] & mel_mask[:, :-1]
        dt_p = mel_pred[:, :, 1:] - mel_pred[:, :, :-1]
        dt_g = mel_gt[:, :, 1:] - mel_gt[:, :, :-1]
        loss_time = (F.l1_loss(dt_p, dt_g, reduction="none") * t_pair_mask[:, None, :]).sum() / (t_pair_mask.sum().clamp(min=1) * n_mels)
    else:
        loss_time = mel_pred.new_tensor(0.0)
    return 0.6 * loss_freq + 0.4 * loss_time


class AetherAcousticModel(nn.Module):
    def __init__(self, vocab_size: int, backbone_type: str = "chunk_attn", d_model: int = 256, n_mels: int = 100, k_left: int = 3, lookahead: int = 8, swa_window: int = 48, chunk_size: int = 4, frame_lookahead: int = 2):
        super().__init__()
        self.backbone_type = backbone_type
        self.d_model = d_model
        self.n_mels = n_mels
        self.lookahead = lookahead
        self.frame_lookahead = frame_lookahead
        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_emb = nn.Embedding(4, d_model)
        self.boundary_emb = nn.Embedding(4, d_model)
        nn.init.normal_(self.boundary_emb.weight, std=0.02)

        self.frame_phase_proj = nn.Linear(FRAME_PHASE_DIM, d_model)
        nn.init.normal_(self.frame_phase_proj.weight, std=0.02)
        nn.init.zeros_(self.frame_phase_proj.bias)

        if backbone_type == "offline_transformer":
            self.backbone = OfflineBiTransformerBackbone(d_model=d_model, n_layers=4, n_heads=4, d_ff=896)
        elif backbone_type == "conv_gru":
            self.backbone = ConvUniGRUBackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_conv=4)
        elif backbone_type == "swa":
            self.backbone = SWABackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_layers=5, n_heads=4, window_size=swa_window)
        elif backbone_type == "chunk_attn":
            self.backbone = ChunkLookaheadBackbone(d_model=d_model, chunk_size=chunk_size, lookahead=lookahead, n_layers=5, n_heads=4, window_size=swa_window)
        elif backbone_type == "gdn":
            self.backbone = GatedDeltaNetBackbone(d_model=d_model, k_left=k_left, lookahead=lookahead, n_layers=5, n_heads=4)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        self.is_streaming = self.backbone.is_streaming
        self.aligner = AlignmentEncoder(n_mels=n_mels, d_text=d_model, d_attn=80)
        self.dur_predictor = CausalPredictorHead(d_model, kernel_size=3, out_dim=1)
        nn.init.constant_(self.dur_predictor.proj.bias, math.log(9.0))
        # 单头联合预测 (f0_norm, uv_logit)：零额外卷积开销补齐清浊音建模
        self.f0_uv_predictor = CausalPredictorHead(d_model, kernel_size=3, out_dim=2)
        with torch.no_grad():
            self.f0_uv_predictor.proj.bias[0] = 0.0
            self.f0_uv_predictor.proj.bias[1] = 1.0
        self.energy_predictor = CausalPredictorHead(d_model, kernel_size=3, out_dim=1)
        self.f0_emb = nn.Linear(1, d_model)
        self.uv_emb = nn.Linear(1, d_model)
        self.energy_emb = nn.Linear(1, d_model)
        self.frame_decoder = nn.ModuleList(
            [CentralizedLookaheadBlock(d_model, k_left=4, lookahead=frame_lookahead, mult=2)]
            + [CausalConvNeXtBlock(d_model, kernel_size=5, mult=2) for _ in range(3)]
        )
        self.mel_out = nn.Linear(d_model, n_mels)

    def embed_text(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, text_mask: torch.Tensor = None, boundary_ids: torch.Tensor = None):
        x_emb = self.phoneme_emb(phoneme_ids) + self.lang_emb(lang_ids)
        if boundary_ids is not None:
            x_emb = x_emb + self.boundary_emb(boundary_ids)
        else:
            x_emb = x_emb + self.boundary_emb(torch.zeros_like(phoneme_ids))
        return x_emb if text_mask is None else x_emb * text_mask[:, :, None].to(x_emb.dtype)

    def encode_text(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, text_mask: torch.Tensor = None, boundary_ids: torch.Tensor = None):
        return self.backbone(self.embed_text(phoneme_ids, lang_ids, text_mask, boundary_ids=boundary_ids), mask=text_mask)

    def _frame_phase_embed(self, durations: torch.Tensor, max_frames: int, dtype: torch.dtype) -> torch.Tensor:
        w = self.frame_phase_proj.weight
        p_dtype = w.dtype if isinstance(w, torch.Tensor) else torch.float32
        feats = build_frame_phase_features(durations, max_frames).to(dtype=p_dtype)
        return self.frame_phase_proj(feats).to(dtype)

    def expand_frames(self, h_txt: torch.Tensor, durations: torch.Tensor, max_frames: int = None):
        h_frame, out_lens = expand_by_duration(h_txt, durations, max_frames=max_frames)
        valid = torch.arange(h_frame.shape[1], device=h_frame.device)[None, :, None] < out_lens[:, None, None]
        h_frame = h_frame + self._frame_phase_embed(durations, h_frame.shape[1], h_frame.dtype)
        return h_frame * valid.to(h_frame.dtype), out_lens

    def _condition_decoder(self, h_frame: torch.Tensor, f0_val: torch.Tensor, uv_prob: torch.Tensor, e_val: torch.Tensor, mel_mask: torch.Tensor = None) -> torch.Tensor:
        # 清音帧 (uv_prob -> 0) 自动屏蔽伪插值 F0，杜绝清擦音/塞擦音声母被谐波污染
        f0_voiced = (f0_val * uv_prob).unsqueeze(-1)
        h_dec = h_frame + self.f0_emb(f0_voiced) + self.uv_emb(uv_prob.unsqueeze(-1)) + self.energy_emb(e_val.unsqueeze(-1))
        if mel_mask is not None:
            h_dec = h_dec * mel_mask[:, :, None].to(h_frame.dtype)
        for blk in self.frame_decoder:
            h_dec = blk(h_dec)
        return self.mel_out(h_dec).transpose(1, 2)

    def forward_train(self, phoneme_ids, lang_ids, text_lens, mel_gt, f0_gt, energy_gt, mel_lens, step_ratio: float = 1.0, boundary_ids: torch.Tensor = None, uv_gt: torch.Tensor = None):
        B, T_max = phoneme_ids.shape
        F_max = mel_gt.shape[-1]
        text_mask = torch.arange(T_max, device=phoneme_ids.device)[None, :] < text_lens[:, None]
        mel_mask = torch.arange(F_max, device=phoneme_ids.device)[None, :] < mel_lens[:, None]
        if uv_gt is None:
            uv_gt = torch.ones_like(f0_gt)

        x_emb = self.embed_text(phoneme_ids, lang_ids, text_mask, boundary_ids=boundary_ids)
        h_txt = self.backbone(x_emb, mask=text_mask)
        attn_log_prior = build_beta_binomial_log_prior(text_lens, mel_lens, T_max, F_max, scaling=1.0)
        attn_logprob, attn_soft = self.aligner(mel_gt, x_emb, text_mask, attn_log_prior=attn_log_prior, mel_mask=mel_mask)
        loss_align = forward_sum_loss(attn_logprob, text_lens, mel_lens)
        attn_hard, dur_gt = viterbi_monotonic_alignment(attn_logprob, text_lens, mel_lens)

        # 平滑软硬对齐退火（0.15 -> 0.30 线性过渡至纯 STE 硬对齐，消除硬跳变震荡）
        attn_ste = attn_hard + (attn_soft - attn_soft.detach())
        if (not self.training) or step_ratio >= 0.30:
            attn_use = attn_ste
        elif step_ratio <= 0.15:
            attn_use = attn_soft
        else:
            alpha = (step_ratio - 0.15) / 0.15
            attn_use = (1.0 - alpha) * attn_soft + alpha * attn_ste

        h_frame = (torch.bmm(attn_use.to(h_txt.dtype), h_txt) + self._frame_phase_embed(dur_gt, F_max, h_txt.dtype)) * mel_mask[:, :, None].to(h_txt.dtype)

        loss_bin = -(attn_hard * attn_logprob * mel_mask[:, :, None]).sum() / mel_mask.sum().clamp(min=1)
        bin_weight = 0.0 if step_ratio < 0.15 else min(1.0, (step_ratio - 0.15) / 0.25)

        # 对 dur_predictor 输入做 0.1x 梯度缩放（防止早期离散 Viterbi 时长噪声冲刷文本主干）
        h_dur_in = h_txt.detach() + 0.1 * (h_txt - h_txt.detach())
        log_dur_pred = self.dur_predictor(h_dur_in)
        loss_dur = (F.l1_loss(log_dur_pred, torch.log(dur_gt.float() + 1.0), reduction="none") * text_mask).sum() / text_mask.sum().clamp(min=1)

        f0_uv_out = self.f0_uv_predictor(h_frame)
        f0_pred, uv_logit = f0_uv_out[..., 0], f0_uv_out[..., 1]
        uv_prob = torch.sigmoid(uv_logit)
        energy_pred = self.energy_predictor(h_frame)

        # 仅在真实浊音帧 (uv_gt > 0.5) 监督 F0，彻底避免强迫模型在静音/清辅音段拟合伪线性插值！
        voiced_mask = mel_mask & (uv_gt > 0.5)
        loss_f0 = (F.l1_loss(f0_pred, f0_gt, reduction="none") * voiced_mask).sum() / voiced_mask.sum().clamp(min=1)
        loss_uv = (F.binary_cross_entropy_with_logits(uv_logit.float(), uv_gt.float(), reduction="none") * mel_mask).sum() / mel_mask.sum().clamp(min=1)
        loss_energy = (F.l1_loss(energy_pred, energy_gt, reduction="none") * mel_mask).sum() / mel_mask.sum().clamp(min=1)

        # 平滑递增预测特征 Scheduled Sampling（0.20 -> 0.70 线性升至 60% 概率；验证期 100% 用预测特征）
        p_sched = 1.0 if (not self.training) else (0.0 if step_ratio < 0.20 else min(0.60, (step_ratio - 0.20) / 0.50 * 0.60))
        use_pred = (not self.training) or (p_sched > 0.0 and torch.rand(1).item() < p_sched)
        f0_in = f0_pred.detach() if use_pred else f0_gt
        uv_in = uv_prob.detach() if use_pred else uv_gt.to(h_frame.dtype)
        e_in = energy_pred.detach() if use_pred else energy_gt

        mel_pred = self._condition_decoder(h_frame, f0_in, uv_in, e_in, mel_mask=mel_mask)
        loss_mel = (F.l1_loss(mel_pred, mel_gt, reduction="none") * mel_mask[:, None, :]).sum() / (mel_mask.sum().clamp(min=1) * self.n_mels)
        loss_struct = mel_structure_loss(mel_pred, mel_gt, mel_mask)

        total_loss = (
            loss_mel
            + 0.35 * loss_struct
            + 0.50 * loss_dur
            + 0.35 * loss_f0
            + 0.25 * loss_uv
            + 0.25 * loss_energy
            + 0.50 * loss_align
            + 0.50 * bin_weight * loss_bin
        )
        bos_ratio = (dur_gt[:, 0].float() / mel_lens.float().clamp(min=1)).mean().detach()
        return {
            "loss": total_loss, "mel": loss_mel.detach(), "struct": loss_struct.detach(),
            "dur": loss_dur.detach(), "f0": loss_f0.detach(), "uv": loss_uv.detach(),
            "energy": loss_energy.detach(), "align": loss_align.detach(),
            "bin": loss_bin.detach(), "bos_ratio": bos_ratio,
        }

    @torch.no_grad()
    def extract_gta_mel(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, text_lens: torch.Tensor, mel_gt: torch.Tensor, mel_lens: torch.Tensor, boundary_ids: torch.Tensor = None) -> torch.Tensor:
        """导出与真实波形逐样本帧对齐的声学预测 Mel (GTA)，供声码器 Stage 2 闭环消除过平滑与分布失配。"""
        B, T_max = phoneme_ids.shape
        F_max = mel_gt.shape[-1]
        text_mask = torch.arange(T_max, device=phoneme_ids.device)[None, :] < text_lens[:, None]
        mel_mask = torch.arange(F_max, device=phoneme_ids.device)[None, :] < mel_lens[:, None]
        x_emb = self.embed_text(phoneme_ids, lang_ids, text_mask, boundary_ids=boundary_ids)
        h_txt = self.backbone(x_emb, mask=text_mask)
        prior = build_beta_binomial_log_prior(text_lens, mel_lens, T_max, F_max, scaling=1.0)
        alp, _ = self.aligner(mel_gt, x_emb, text_mask, attn_log_prior=prior, mel_mask=mel_mask)
        attn_hard, dur_gt = viterbi_monotonic_alignment(alp, text_lens, mel_lens)
        h_frame = (torch.bmm(attn_hard.to(h_txt.dtype), h_txt) + self._frame_phase_embed(dur_gt, F_max, h_txt.dtype)) * mel_mask[:, :, None].to(h_txt.dtype)
        f0_uv_out = self.f0_uv_predictor(h_frame)
        f0_pred, uv_prob = f0_uv_out[..., 0], torch.sigmoid(f0_uv_out[..., 1])
        e_pred = self.energy_predictor(h_frame)
        return self._condition_decoder(h_frame, f0_pred, uv_prob, e_pred, mel_mask=mel_mask)

    @torch.no_grad()
    def infer_offline(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, speed: float = 1.0, boundary_ids: torch.Tensor = None):
        h_txt = self.encode_text(phoneme_ids, lang_ids, boundary_ids=boundary_ids)
        dur = torch.clamp(torch.round((torch.exp(self.dur_predictor(h_txt)) - 1.0) / speed), min=1, max=60).long()
        h_frame, _ = self.expand_frames(h_txt, dur)
        f0_uv = self.f0_uv_predictor(h_frame)
        f0_pred, uv_prob = f0_uv[..., 0], torch.sigmoid(f0_uv[..., 1])
        e_pred = self.energy_predictor(h_frame)
        mel_out = self._condition_decoder(h_frame, f0_pred, uv_prob, e_pred, mel_mask=None)
        return mel_out, dur, f0_pred, e_pred

    def init_stream_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        assert self.is_streaming
        device = device or next(self.parameters()).device
        return {
            "backbone": self.backbone.init_state(batch_size, device, dtype),
            "dur": self.dur_predictor.init_state(batch_size, device, dtype),
            "f0_uv": self.f0_uv_predictor.init_state(batch_size, device, dtype),
            "energy": self.energy_predictor.init_state(batch_size, device, dtype),
            "frame_dec": [blk.init_state(batch_size, device, dtype) for blk in self.frame_decoder],
        }

    @torch.no_grad()
    def infer_stream_step(self, phoneme_chunk: torch.Tensor, lang_chunk: torch.Tensor, state: dict, is_final: bool = False, speed: float = 1.0, boundary_chunk: torch.Tensor = None):
        if phoneme_chunk.shape[1] > 0:
            b_chunk = boundary_chunk if boundary_chunk is not None else torch.zeros_like(phoneme_chunk)
            x = self.phoneme_emb(phoneme_chunk) + self.lang_emb(lang_chunk) + self.boundary_emb(b_chunk)
        else:
            x = torch.zeros(phoneme_chunk.shape[0], 0, self.d_model, device=phoneme_chunk.device, dtype=self.phoneme_emb.weight.dtype)
        h_emit, state["backbone"] = self.backbone.forward_step(x, state["backbone"], is_final=is_final)
        if h_emit.shape[1] == 0:
            if not is_final:
                return torch.zeros(x.shape[0], self.n_mels, 0, device=x.device, dtype=x.dtype), state
            h_dec = torch.zeros(x.shape[0], 0, self.d_model, device=x.device, dtype=x.dtype)
        else:
            log_dur, state["dur"] = self.dur_predictor.forward_step(h_emit, state["dur"])
            dur = torch.clamp(torch.round((torch.exp(log_dur) - 1.0) / speed), min=1, max=60).long()
            h_frame, _ = self.expand_frames(h_emit, dur)
            f0_uv, state["f0_uv"] = self.f0_uv_predictor.forward_step(h_frame, state["f0_uv"])
            f0_pred, uv_prob = f0_uv[..., 0], torch.sigmoid(f0_uv[..., 1])
            e_pred, state["energy"] = self.energy_predictor.forward_step(h_frame, state["energy"])
            h_dec = h_frame + self.f0_emb((f0_pred * uv_prob).unsqueeze(-1)) + self.uv_emb(uv_prob.unsqueeze(-1)) + self.energy_emb(e_pred.unsqueeze(-1))
        h_dec, f0_st = self.frame_decoder[0].forward_step(h_dec, state["frame_dec"][0], is_final=is_final)
        new_fdec = [f0_st]
        for blk, fst in zip(self.frame_decoder[1:], state["frame_dec"][1:]):
            h_dec, nfst = blk.forward_step(h_dec, fst)
            new_fdec.append(nfst)
        state["frame_dec"] = new_fdec
        if h_dec.shape[1] == 0:
            return torch.zeros(x.shape[0], self.n_mels, 0, device=x.device, dtype=x.dtype), state
        return self.mel_out(h_dec).transpose(1, 2), state

    def export_inference_copy(self) -> nn.Module:
        """克隆纯推理解耦副本并剥离仅训练期使用的 aligner，确保端侧导出/INT8 体积精确无冗余。"""
        m_inf = copy.deepcopy(self).to("cpu").eval()
        m_inf.aligner = nn.Identity()
        return m_inf


# ==============================================================================
# 5.4 24 kHz 帧率谐波源因果声码器 CausalVocosLite (a5) + 修复版转置 SpecDisc & PeriodDisc(11)
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
        out = F.fold(
            frames.transpose(1, 2),
            output_size=(1, total_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).reshape(B, total_len)
        w2 = (self.window ** 2).view(1, self.n_fft, 1).expand(1, self.n_fft, n_frames)
        env = F.fold(
            w2,
            output_size=(1, total_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).reshape(1, total_len)
        left_trim = self.n_fft // 2
        return out[:, left_trim : left_trim + n_frames * self.hop_length] / env[:, left_trim : left_trim + n_frames * self.hop_length].clamp(min=1e-6)

    def init_state(self, batch_size: int, device):
        pad = self.n_fft - self.hop_length
        return {
            "ola_out": torch.zeros(batch_size, pad, device=device, dtype=torch.float32),
            "ola_env": torch.zeros(batch_size, pad, device=device, dtype=torch.float32),
            "trim_remaining": self.n_fft // 2,
        }

    @torch.no_grad()
    def forward_step(self, x_chunk: torch.Tensor, state: dict, is_final: bool = False):
        B, C, _ = x_chunk.shape
        pad = self.n_fft - self.hop_length
        left_trim = self.n_fft // 2
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
            tail_end = emit_end + left_trim
            wav_emit = torch.zeros(B, 0, device=x_chunk.device, dtype=torch.float32) if tail_end <= trim_rem else out[:, trim_rem:tail_end] / env[:, trim_rem:tail_end].clamp(min=1e-6)
            return wav_emit, self.init_state(B, x_chunk.device)
        if emit_end <= trim_rem:
            wav_emit = torch.zeros(B, 0, device=x_chunk.device, dtype=torch.float32)
            new_trim = trim_rem - emit_end
        else:
            wav_emit = out[:, trim_rem:emit_end] / env[:, trim_rem:emit_end].clamp(min=1e-6)
            new_trim = 0
        return wav_emit, {"ola_out": out[:, emit_end : emit_end + pad].clone(), "ola_env": env[:, emit_end : emit_end + pad].clone(), "trim_remaining": new_trim}


class CausalHarmonicSource(nn.Module):
    """帧率 (93.75 Hz) 因果谐波相位振荡器：
    当元音持续数十帧 Mel 近似恒定时，纯局部卷积无法维持跨帧连续旋转的 STFT 相位，必出 93.75Hz 调制电音。
    本模块从带 2 帧前瞻的声学表示预测逐帧 (log_f0, uv_logit)，在 float64 下做 mod-1 包裹基频相位积分并合成 12 阶谐波正余弦参考，
    零 24kHz 采样率开销、无限流式零漂移、离线与任意切块流式 100% 逐位数学等价。"""

    def __init__(self, dim: int = 256, hop_length: int = 256, sample_rate: int = 24000, n_harmonics: int = 12):
        super().__init__()
        self.dim = dim
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        hid = dim // 2
        self.pitch_conv = CausalConv1d(dim, hid, kernel_size=3)
        self.pitch_norm = nn.LayerNorm(hid)
        self.pitch_out = nn.Linear(hid, 2)
        with torch.no_grad():
            self.pitch_out.bias[0] = math.log(210.0)
            self.pitch_out.bias[1] = 1.0
        feat_dim = n_harmonics * 2 + 2
        self.in_proj = nn.Linear(feat_dim, dim)
        self.head_proj = nn.Linear(feat_dim, dim)
        nn.init.normal_(self.in_proj.weight, std=0.02); nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.head_proj.weight, std=0.02); nn.init.zeros_(self.head_proj.bias)
        self.register_buffer("harm_orders", torch.arange(1, n_harmonics + 1, dtype=torch.float64)[None, None, :], persistent=False)

    def _oscillate(self, log_f0_pred: torch.Tensor, uv_logit: torch.Tensor, out_dtype: torch.dtype, phi_init: torch.Tensor = None):
        uv_prob = torch.sigmoid(uv_logit)
        f0_hz = torch.exp(log_f0_pred.float().clamp(min=math.log(55.0), max=math.log(620.0)))
        delta_phi = f0_hz.double() * (float(self.hop_length) / float(self.sample_rate))
        csum = torch.cumsum(delta_phi, dim=1)
        if phi_init is not None:
            csum = phi_init + csum
        phi = csum.fmod(1.0)
        new_phi = phi[:, -1:] if phi.shape[1] > 0 else (phi_init if phi_init is not None else torch.zeros(log_f0_pred.shape[0], 1, device=log_f0_pred.device, dtype=torch.float64))
        angles = (2.0 * math.pi * phi.unsqueeze(-1) * self.harm_orders).to(torch.float32)
        harms = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) * uv_prob.float().unsqueeze(-1)
        f0_norm = ((log_f0_pred.float() - 5.3) / 0.5).unsqueeze(-1)
        w = self.in_proj.weight
        p_dtype = w.dtype if isinstance(w, torch.Tensor) else torch.float32
        feats = torch.cat([harms, f0_norm, uv_prob.float().unsqueeze(-1)], dim=-1).to(p_dtype)
        return self.in_proj(feats).to(out_dtype), self.head_proj(feats).to(out_dtype), new_phi

    def forward(self, h: torch.Tensor):
        p_hid = F.silu(self.pitch_norm(self.pitch_conv(h.transpose(1, 2)).transpose(1, 2)))
        p_out = self.pitch_out(p_hid)
        log_f0_pred, uv_logit = p_out[..., 0], p_out[..., 1]
        harm_in, harm_head, _ = self._oscillate(log_f0_pred, uv_logit, h.dtype, phi_init=None)
        return harm_in, harm_head, log_f0_pred, uv_logit

    def init_state(self, batch_size: int, device, dtype):
        return {
            "conv": self.pitch_conv.init_state(batch_size, device, dtype),
            "phi": torch.zeros(batch_size, 1, device=device, dtype=torch.float64),
        }

    def forward_step(self, h_chunk: torch.Tensor, state: dict):
        B, C, D = h_chunk.shape
        if C == 0:
            empty = torch.zeros(B, 0, D, device=h_chunk.device, dtype=h_chunk.dtype)
            return empty, empty, state
        p_conv, new_conv = self.pitch_conv.forward_step(h_chunk.transpose(1, 2), state["conv"])
        p_hid = F.silu(self.pitch_norm(p_conv.transpose(1, 2)))
        p_out = self.pitch_out(p_hid)
        log_f0_pred, uv_logit = p_out[..., 0], p_out[..., 1]
        harm_in, harm_head, new_phi = self._oscillate(log_f0_pred, uv_logit, h_chunk.dtype, phi_init=state["phi"])
        return harm_in, harm_head, {"conv": new_conv, "phi": new_phi}


class CausalVocosLite(nn.Module):
    def __init__(self, n_mels: int = 100, dim: int = 256, n_layers: int = 8, kernel_size: int = 7, n_fft: int = 1024, hop_length: int = 256, lookahead: int = 2, sample_rate: int = 24000):
        super().__init__()
        self.dim = dim
        # 内置 Mel 归一化统计缓存：消除 LookaheadConv1d 首尾补 0 在原始 log-mel 下造成的 +16 能量爆炸，同时保持外部 raw-mel 接口不变
        self.register_buffer("mel_mean", torch.zeros(1, n_mels, 1, dtype=torch.float32))
        self.register_buffer("mel_std", torch.ones(1, n_mels, 1, dtype=torch.float32))
        self.embed = LookaheadConv1d(n_mels, dim, k_left=kernel_size - 1, lookahead=lookahead)
        self.norm_in = nn.LayerNorm(dim)
        self.harm_source = CausalHarmonicSource(dim=dim, hop_length=hop_length, sample_rate=sample_rate, n_harmonics=12)
        self.blocks = nn.ModuleList([CausalConvNeXtBlock(dim, kernel_size=kernel_size, mult=3) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = StreamingISTFTHead(dim, n_fft=n_fft, hop_length=hop_length)

    def set_mel_stats(self, mel_mean, mel_std):
        with torch.no_grad():
            m = torch.as_tensor(mel_mean, dtype=torch.float32, device=self.mel_mean.device).view(1, -1, 1)
            s = torch.as_tensor(mel_std, dtype=torch.float32, device=self.mel_std.device).view(1, -1, 1).clamp(min=1e-4)
            self.mel_mean.copy_(m)
            self.mel_std.copy_(s)

    def _norm_input(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mel_mean.to(mel.dtype)) / self.mel_std.to(mel.dtype)

    def forward_with_aux(self, mel: torch.Tensor):
        x = self._norm_input(mel)
        h = self.norm_in(self.embed(x).transpose(1, 2))
        harm_in, harm_head, log_f0_pred, uv_logit = self.harm_source(h)
        h = h + harm_in
        for blk in self.blocks:
            h = blk(h)
        wav = self.head(self.norm_out(h + harm_head))
        return wav, log_f0_pred, uv_logit

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(mel)[0]

    def init_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        device = device or next(self.parameters()).device
        return {
            "embed": self.embed.init_state(batch_size, device, dtype),
            "harm": self.harm_source.init_state(batch_size, device, dtype),
            "blocks": [b.init_state(batch_size, device, dtype) for b in self.blocks],
            "head": self.head.init_state(batch_size, device),
        }

    @torch.no_grad()
    def forward_step(self, mel_chunk: torch.Tensor, state: dict, is_final: bool = False):
        x = self._norm_input(mel_chunk) if mel_chunk.shape[-1] > 0 else mel_chunk
        h_emb, state["embed"] = self.embed.forward_step(x, state["embed"], is_final=is_final)
        if h_emb.shape[-1] > 0:
            h = self.norm_in(h_emb.transpose(1, 2))
        else:
            h = torch.zeros(mel_chunk.shape[0], 0, self.dim, device=mel_chunk.device, dtype=mel_chunk.dtype)
        harm_in, harm_head, state["harm"] = self.harm_source.forward_step(h, state["harm"])
        h = h + harm_in
        new_blks = []
        for blk, bst in zip(self.blocks, state["blocks"]):
            h, nb = blk.forward_step(h, bst)
            new_blks.append(nb)
        state["blocks"] = new_blks
        h_out = self.norm_out(h + harm_head) if h.shape[1] > 0 else h
        wav, state["head"] = self.head.forward_step(h_out, state["head"], is_final=is_final)
        return wav, state


def _apply_wn(module: nn.Module) -> nn.Module:
    wn = getattr(nn.utils.parametrizations, "weight_norm", None) or nn.utils.weight_norm
    return wn(module)


class PeriodDisc(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            _apply_wn(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            _apply_wn(nn.Conv2d(32, 64, (5, 1), (3, 1), (2, 0))),
            _apply_wn(nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0))),
            _apply_wn(nn.Conv2d(128, 256, (5, 1), (3, 1), (2, 0))),
            _apply_wn(nn.Conv2d(256, 1, (3, 1), 1, (1, 0))),
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
    """修复版多分辨率频谱判别器 (UnivNet / Vocos MRD 标准布局)：
    - 将 STFT 转置为 (B, 1, time_frames, freq_bins)，使核 (3, 9) 与步长 (1, 2) 沿频率轴跨 73 个 bin 聚合谐波簇！
    - 采用幂律幅度压缩 pow(0.3)，防止低频高能量元音掩盖 2-8 kHz 齿音与自然气声梯度。"""

    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft))
        self.convs = nn.ModuleList([
            _apply_wn(nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4))),
            _apply_wn(nn.Conv2d(32, 64, (3, 9), (1, 2), (1, 4))),
            _apply_wn(nn.Conv2d(64, 128, (3, 9), (1, 2), (1, 4))),
            _apply_wn(nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))),
            _apply_wn(nn.Conv2d(128, 1, (3, 3), (1, 1), (1, 1))),
        ])

    def forward(self, x: torch.Tensor):
        spec = torch.stft(x.float(), self.n_fft, self.hop, self.n_fft, self.win, return_complex=True).abs()
        h = spec.clamp(min=1e-6).pow(0.3).transpose(1, 2).unsqueeze(1).to(x.dtype)
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
            PeriodDisc(2), PeriodDisc(3), PeriodDisc(5), PeriodDisc(7), PeriodDisc(11),
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
