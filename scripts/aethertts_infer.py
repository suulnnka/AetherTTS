#!/usr/bin/env python3
"""
AetherTTS (a9) 推理入口：加载已训练的声学模型 + CausalVocosLite 声码器，文本 -> 24 kHz 单声道波形。

直接 import scripts/aethertts_lib.py（不依赖 Notebook / 训练流程）。checkpoint 自带 cfg / stats / vocab，
推理路径与 aethertts_phase1.py 评测期的 synthesize_offline / synthesize_streaming 完全一致（FP32 + 关闭 TF32）。

命令行：
    python scripts/aethertts_infer.py --list                                  # 列出 <root> 下可用的 checkpoint
    python scripts/aethertts_infer.py --text "你好，请问今天北京的天气怎么样？" --out hello.wav
    python scripts/aethertts_infer.py --text "..." --out hello.wav --stream   # 流式路径，打印首包延迟 / RTF
    python scripts/aethertts_infer.py --text "..." --device cpu --root ./aethertts_exp

Python API（在仓库根目录下运行，或先把仓库根目录加入 sys.path）：
    from scripts.aethertts_infer import AetherTTS

    tts = AetherTTS(root="./aethertts_exp")                  # 自动发现最新 checkpoint；有 CUDA 则用 CUDA
    wav = tts.synthesize("你好，请问今天北京的天气怎么样？")    # np.float32 一维数组，24 kHz
    soundfile.write("hello.wav", wav, tts.sample_rate)

    for chunk in tts.stream("你好，请问今天北京的天气怎么样？"):  # 流式：逐块产出波形，可边收边播
        ...

    sess = tts.open_stream()                                # 增量喂文本（如 LLM 按句读输出），状态跨段不重置
    for piece in ["你好，", "请问今天北京的天气怎么样？"]:
        for chunk in sess.push(piece):
            ...
    for chunk in sess.close():
        ...
"""
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import aethertts_lib as lib

__all__ = ["AetherTTS", "StreamSession", "find_acoustic_ckpt", "find_vocoder_ckpt"]


@contextmanager
def _no_tf32():
    """评测期口径：合成期间关闭 CUDA TF32，避免离线整句卷积与流式分块卷积的数值截断漂移。"""
    on_cuda = torch.cuda.is_available()
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        if on_cuda:
            torch.backends.cuda.matmul.allow_tf32 = prev_mm
            torch.backends.cudnn.allow_tf32 = prev_cudnn


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # 仅用于自己训练产出的 checkpoint：旧断点含 weights_only 白名单外的对象时回退完整反序列化
        return torch.load(path, map_location="cpu", weights_only=False)


def find_acoustic_ckpt(root, explicit: str = "") -> Path:
    """声学 checkpoint：runs/*/best.pt（无则 latest.pt），多个时取最近写入的一个。"""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"声学 checkpoint 不存在: {p}")
        return p
    runs_dir = Path(root).expanduser().resolve() / "runs"
    cands = []
    for run_dir in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        for fn in ("best.pt", "latest.pt"):
            if (run_dir / fn).exists():
                cands.append(run_dir / fn)
                break
    if not cands:
        raise FileNotFoundError(f"{runs_dir} 下没有 best.pt / latest.pt：请先训练，或用 --root / --acoustic-ckpt 指定位置")
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_vocoder_ckpt(root, ck: dict, explicit: str = "") -> Path:
    """与声学 checkpoint 配套的声码器：vocoder/causal_vocos_lite_<run_suffix>_L<vocoder_lookahead>/final.pt（无则 latest.pt）。"""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"声码器 checkpoint 不存在: {p}")
        return p
    cfg = ck["cfg"]
    run_suffix = cfg.get("run_suffix") or f"{cfg['dataset_source']}_{cfg['profile']}_{ck.get('arch_tag', '')}"
    voc_dir = Path(root).expanduser().resolve() / "vocoder" / f"causal_vocos_lite_{run_suffix}_L{cfg['vocoder_lookahead']}"
    for fn in ("final.pt", "latest.pt"):
        if (voc_dir / fn).exists():
            return voc_dir / fn
    raise FileNotFoundError(f"{voc_dir} 下没有 final.pt / latest.pt：声码器尚未训练完成，或用 --vocoder-ckpt 指定位置")


class StreamSession:
    """一条不重置状态的流式合成会话：文本按段 push（每段内部按 stream_text_chunk 个音素分块喂入声学模型），
    声码器首包攒够 stream_mel_first_chunk 帧即出声、稳态每 stream_mel_chunk 帧出一块；close() 补 <eos> 并冲刷前瞻与 OLA 尾部。"""

    def __init__(self, tts: "AetherTTS", speed: float = 1.0):
        self._tts = tts
        self._speed = speed
        fe = tts.frontend
        self._bos, self._eos = fe.bos_id, fe.eos_id
        self._pending, self._started, self._closed, self._n_phonemes = [], False, False, 0
        self._ast = tts.acoustic.init_stream_state(1, device=tts.device, dtype=torch.float32)
        self._voc = tts.vocoder.init_state(1, device=tts.device, dtype=torch.float32)
        self._mel_buf = torch.zeros(1, tts.cfg["n_mels"], 0, device=tts.device, dtype=torch.float32)
        self._first_emitted = False

    def push(self, text: str) -> list:
        """喂入一段文本，返回本段新产出的波形块列表（可能为空：需等后续文本凑满前瞻才出声）。"""
        if self._closed:
            raise RuntimeError("会话已 close，请重新 open_stream()")
        return list(self._run([text], final=False))

    def close(self) -> list:
        """结束这条流：<eos> 入队并冲刷声学前瞻与声码器 OLA 尾部，返回剩余波形块。"""
        if self._closed:
            raise RuntimeError("会话已 close，请重新 open_stream()")
        self._closed = True
        return list(self._run([], final=True))

    def _run(self, texts, final: bool, require_speech: bool = False):
        t_step = self._tts.cfg["stream_text_chunk"]
        with torch.no_grad(), _no_tf32():
            for text in texts:
                pids, lids, bids = self._tts.frontend.text_to_ids(text)
                if not self._started:  # <bos>/<eos> 只属于整条流的首尾，句中各段去掉
                    self._pending.append((self._bos, 0, 0))
                    self._started = True
                self._pending.extend(zip(pids[1:-1], lids[1:-1], bids[1:-1]))
                self._n_phonemes += max(len(pids) - 2, 0)
            if final and require_speech and self._n_phonemes == 0:
                raise ValueError("文本中没有可发音的内容")
            if final:
                if not self._started:
                    self._pending.append((self._bos, 0, 0))
                    self._started = True
                self._pending.append((self._eos, 0, 0))
            while len(self._pending) > t_step:
                chunk, self._pending = self._pending[:t_step], self._pending[t_step:]
                yield from self._step(chunk, is_final=False)
            if final:
                chunk, self._pending = self._pending, []
                yield from self._step(chunk, is_final=True)
                wc, self._voc = self._tts.vocoder.forward_step(self._mel_buf, self._voc, is_final=True)
                self._mel_buf = self._mel_buf[:, :, :0]
                if wc.shape[-1] > 0:
                    yield wc.squeeze(0).float().cpu().numpy()

    def _step(self, chunk, is_final: bool):
        dev = self._tts.device
        pt = torch.tensor([[t[0] for t in chunk]], dtype=torch.long, device=dev)
        lt = torch.tensor([[t[1] for t in chunk]], dtype=torch.long, device=dev)
        bt = torch.tensor([[t[2] for t in chunk]], dtype=torch.long, device=dev)
        mc, self._ast = self._tts.acoustic.infer_stream_step(
            pt, lt, self._ast, is_final=is_final, speed=self._speed, boundary_chunk=bt,
        )
        if mc.shape[-1] > 0:
            self._mel_buf = torch.cat([self._mel_buf, self._tts._denorm(mc)], dim=-1)
        cfg = self._tts.cfg
        while True:
            m_step = cfg["stream_mel_first_chunk"] if not self._first_emitted else cfg["stream_mel_chunk"]
            if self._mel_buf.shape[-1] < m_step:
                break
            wc, self._voc = self._tts.vocoder.forward_step(self._mel_buf[:, :, :m_step], self._voc, is_final=False)
            self._mel_buf = self._mel_buf[:, :, m_step:]
            if wc.shape[-1] > 0:
                self._first_emitted = True
                yield wc.squeeze(0).float().cpu().numpy()


class AetherTTS:
    """文本 -> 波形。synthesize() 为整句离线；stream() / open_stream() 为流式（与离线输出数值一致）。"""

    def __init__(self, root="./aethertts_exp", device=None, acoustic_ckpt=None, vocoder_ckpt=None):
        root = Path(root).expanduser().resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ac_path = find_acoustic_ckpt(root, acoustic_ckpt)
        ck = _torch_load(ac_path)
        cfg = ck["cfg"]
        if list(ck["vocab"]) != lib.FRONTEND.vocab:
            raise RuntimeError(
                f"aethertts_lib.py 前端词表与 checkpoint 不一致: {ac_path}\n"
                "请先运行 python scripts/sync_phase1_script.py 重新同步，或指定匹配的 --acoustic-ckpt"
            )
        self.cfg = cfg
        self.sample_rate = int(cfg["sample_rate"])
        self.frontend = lib.FRONTEND
        self.acoustic_path, self.arch_tag, self.step = ac_path, ck.get("arch_tag"), ck.get("step")

        # 结构参数全部取自 checkpoint 自带的 cfg，不受 AETHER_* 环境变量影响
        self.acoustic = lib.AetherAcousticModel(
            vocab_size=len(ck["vocab"]), backbone_type=ck["backbone_type"], d_model=cfg["d_model"],
            n_mels=cfg["n_mels"], k_left=cfg["k_left"], lookahead=ck["lookahead"],
            swa_window=cfg["swa_window"], chunk_size=cfg["chunk_attn_size"],
            frame_lookahead=cfg["frame_lookahead"], dropout=cfg["dropout_acoustic"],
        )
        self.acoustic.load_state_dict(ck["model"])

        self.vocoder_path = find_vocoder_ckpt(root, ck, vocoder_ckpt)
        voc_sd = _torch_load(self.vocoder_path)
        if "voc" in voc_sd:  # latest.pt 是含优化器 / 判别器状态的训练断点
            voc_sd = voc_sd["voc"]
        self.vocoder = lib.CausalVocosLite(
            n_mels=cfg["n_mels"], dim=cfg["vocoder_dim"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
            lookahead=cfg["vocoder_lookahead"], sample_rate=cfg["sample_rate"],
        )
        self.vocoder.load_state_dict(voc_sd)  # 含 mel_mean / mel_std 缓冲

        self.acoustic.to(self.device).eval()
        self.vocoder.to(self.device).eval()
        stats = ck["stats"]
        self._mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32, device=self.device)[None, :, None]
        self._mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32, device=self.device)[None, :, None]

    def _denorm(self, mel_norm: torch.Tensor) -> torch.Tensor:
        return mel_norm * self._mel_std + self._mel_mean

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """整句离线合成，返回 float32 波形（24 kHz 单声道，幅度未削波，写盘前可 clip 到 [-1, 1]）。"""
        pids, lids, bids = self.frontend.text_to_ids(text)
        if len(pids) <= 2:
            raise ValueError(f"文本中没有可发音的内容: {text!r}")
        with torch.no_grad(), _no_tf32():
            mel_norm, _, _, _ = self.acoustic.infer_offline(
                torch.tensor([pids], dtype=torch.long, device=self.device),
                torch.tensor([lids], dtype=torch.long, device=self.device),
                speed=speed, boundary_ids=torch.tensor([bids], dtype=torch.long, device=self.device),
            )
            wav = self.vocoder(self._denorm(mel_norm))
        return wav.squeeze(0).float().cpu().numpy()

    def stream(self, text: str, speed: float = 1.0):
        """整句走流式路径，逐块 yield 波形（生成器；空文本在首次迭代时抛 ValueError）。"""
        return self.open_stream(speed)._run([text], final=True, require_speech=True)

    def open_stream(self, speed: float = 1.0) -> StreamSession:
        """开启增量流式会话：按标点切分、逐段 push，状态跨段不重置。"""
        return StreamSession(self, speed=speed)

    def warmup(self, text: str = "预热。"):
        """提前触发前端词典加载与设备内核预热，避免首个真实请求承担冷启动耗时（英文 G2P 需传入含英文的 text）。"""
        self.synthesize(text)
        for _ in self.stream(text):
            pass


def main():
    ap = argparse.ArgumentParser(description="AetherTTS (a9) inference: text -> wav.")
    ap.add_argument("--text", help="要合成的文本（中英混读均可；#1/#2/#3 可手工标注韵律边界）.")
    ap.add_argument("--out", default="aethertts_out.wav", help="输出 wav 路径 (default: aethertts_out.wav).")
    ap.add_argument("--root", default="./aethertts_exp", help="训练产物目录 (datasets / runs / vocoder).")
    ap.add_argument("--device", default=None, help="cuda / cpu (默认有 CUDA 则用 CUDA).")
    ap.add_argument("--acoustic-ckpt", default=None, help="声学 checkpoint (默认自动发现最新 best.pt).")
    ap.add_argument("--vocoder-ckpt", default=None, help="声码器 checkpoint (默认自动匹配配套 final.pt).")
    ap.add_argument("--speed", type=float, default=1.0, help="语速倍率 (>1 更快).")
    ap.add_argument("--stream", action="store_true", help="走流式路径并打印首包延迟 / RTF.")
    ap.add_argument("--list", action="store_true", help="列出可用 checkpoint 后退出.")
    args = ap.parse_args()

    if args.list:
        root = Path(args.root).expanduser().resolve()
        for sub, fns in (("runs", ("best.pt", "latest.pt")), ("vocoder", ("final.pt", "latest.pt"))):
            for d in sorted((root / sub).glob("*")) if (root / sub).exists() else []:
                found = [f"{fn} ({time.strftime('%Y-%m-%d %H:%M', time.localtime((d / fn).stat().st_mtime))})"
                         for fn in fns if (d / fn).exists()]
                if found:
                    print(f"{sub}/{d.name}: {', '.join(found)}")
        return
    if not args.text:
        ap.error("需要 --text（或用 --list 查看 checkpoint）")

    tts = AetherTTS(root=args.root, device=args.device, acoustic_ckpt=args.acoustic_ckpt, vocoder_ckpt=args.vocoder_ckpt)
    print(f"[Load] arch={tts.arch_tag} step={tts.step} | device={tts.device}\n"
          f"       acoustic: {tts.acoustic_path}\n       vocoder : {tts.vocoder_path}")
    tts.warmup()  # 计时不含前端词典加载 / 设备内核预热等一次性冷启动

    t0 = time.perf_counter()
    if args.stream:
        chunks, first_ms = [], None
        for c in tts.stream(args.text, speed=args.speed):
            if first_ms is None and len(c) > 0:
                first_ms = (time.perf_counter() - t0) * 1000.0
            chunks.append(c)
        wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        print(f"[Stream] {len(chunks)} 块 | 首包 {first_ms if first_ms is not None else float('nan'):.1f} ms | 首块 {len(chunks[0]) / tts.sample_rate * 1000 if chunks else 0:.1f} ms")
    else:
        wav = tts.synthesize(args.text, speed=args.speed)
    elapsed = time.perf_counter() - t0

    wav = np.clip(wav, -1.0, 1.0)
    sf.write(args.out, wav, tts.sample_rate)
    dur = len(wav) / tts.sample_rate
    print(f"[Saved] {args.out} | 时长 {dur:.2f} s | 合成耗时 {elapsed * 1000:.0f} ms (RTF {elapsed / max(dur, 1e-6):.3f})")


if __name__ == "__main__":
    main()
