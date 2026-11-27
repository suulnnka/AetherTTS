#!/usr/bin/env python3
"""
AetherTTS 轻量纯 CPU ONNX Runtime Python 推理脚本（零 PyTorch 依赖）。

仅依赖 numpy + onnxruntime + pypinyin（可选 jieba / g2p_en / soundfile），
支持直接加载 scripts/export_onnx.py 导出的离线两段式与三段流式 ONNX 模型并合成 24 kHz WAV。

用法示例:
    # 1) 离线推理（默认）
    python scripts/infer_onnx.py --onnx-dir ./onnx_export \\
        --text "你好，请问今天北京的天气怎么样？" --out out.wav

    # 2) 三段式真流式推理（实时逐块吐出 PCM，并统计首包延迟 TTFA 与 RTF）
    python scripts/infer_onnx.py --onnx-dir ./onnx_export --mode streaming \\
        --text "你好，请问今天北京的天气怎么样？" --out out_stream.wav

    # 3) 使用 INT8 量化模型 + 2 线程 CPU + 1.1x 语速
    python scripts/infer_onnx.py --onnx-dir ./onnx_export --int8 --threads 2 --speed 1.1 \\
        --text "帮我把 Wi-Fi 切换到 5G，一会儿见！" --out out_int8.wav
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.text_frontend import BilingualPhonemeFrontend


def _create_session(onnx_path: Path, num_threads: int = 2) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _resolve_model(onnx_dir: Path, stem: str, use_int8: bool) -> Path:
    if use_int8:
        cand = onnx_dir / f"{stem}.int8.onnx"
        if cand.exists():
            return cand
        print(f"[Warn] 未找到 {cand.name}，自动回退到 FP32 模型 {stem}.onnx")
    return onnx_dir / f"{stem}.onnx"


def save_wav(path: Path, wav: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    try:
        import soundfile as sf

        sf.write(str(path), wav, sample_rate)
    except ImportError:
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())


# ==============================================================================
# 1. 离线两段式推理引擎 (acoustic.onnx -> vocoder.onnx)
# ==============================================================================
class AetherONNXOfflineEngine:
    def __init__(self, onnx_dir: Path, use_int8: bool = False, num_threads: int = 2):
        self.onnx_dir = Path(onnx_dir).expanduser().resolve()
        self.meta = json.loads((self.onnx_dir / "onnx_meta.json").read_text(encoding="utf-8"))
        self.sample_rate = int(self.meta["sample_rate"])
        self.frontend = BilingualPhonemeFrontend(vocab=self.meta.get("vocab"))
        self.acoustic_sess = _create_session(_resolve_model(self.onnx_dir, "acoustic", use_int8), num_threads)
        self.vocoder_sess = _create_session(_resolve_model(self.onnx_dir, "vocoder", use_int8), num_threads)

    def synthesize_ids(self, phoneme_ids: list, lang_ids: list, boundary_ids: list, speed: float = 1.0):
        t0 = time.perf_counter()
        mel, durations = self.acoustic_sess.run(
            None,
            {
                "phoneme_ids": np.asarray([phoneme_ids], dtype=np.int64),
                "lang_ids": np.asarray([lang_ids], dtype=np.int64),
                "boundary_ids": np.asarray([boundary_ids], dtype=np.int64),
                "speed": np.asarray([speed], dtype=np.float32),
            },
        )
        t_ac = time.perf_counter() - t0

        t1 = time.perf_counter()
        (wav,) = self.vocoder_sess.run(None, {"mel": mel})
        t_voc = time.perf_counter() - t1

        wav_1d = wav.squeeze(0).astype(np.float32)
        dur_sec = max(len(wav_1d) / float(self.sample_rate), 1e-6)
        total_sec = t_ac + t_voc
        stats = {
            "n_phonemes": len(phoneme_ids),
            "n_frames": int(mel.shape[-1]),
            "audio_sec": dur_sec,
            "acoustic_ms": t_ac * 1000.0,
            "vocoder_ms": t_voc * 1000.0,
            "total_ms": total_sec * 1000.0,
            "rtf": total_sec / dur_sec,
        }
        return wav_1d, mel, durations, stats

    def synthesize(self, text: str, speed: float = 1.0):
        pids, lids, bids = self.frontend.text_to_ids(text)
        return self.synthesize_ids(pids, lids, bids, speed=speed)


# ==============================================================================
# 2. 三段式流式推理引擎 (text_step -> frame_step -> vocoder_step + OLA)
# ==============================================================================
class StreamingOLA:
    """与 StreamingISTFTHead.forward_step 逐采样等价的 FP32 流式重叠相加 (OLA) 缓冲。"""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = n_fft - hop_length
        self.left_trim = n_fft // 2
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float64) / float(n_fft)))
        self.w2 = (win.astype(np.float32)) ** 2
        self.reset()

    def reset(self):
        self.ola_out = np.zeros(self.pad, dtype=np.float32)
        self.ola_env = np.zeros(self.pad, dtype=np.float32)
        self.trim_remaining = self.left_trim

    def push(self, time_frames: np.ndarray, is_final: bool = False) -> np.ndarray:
        # time_frames: [C, n_fft]
        C = time_frames.shape[0]
        if C == 0 and not is_final:
            return np.zeros(0, dtype=np.float32)
        out_len = C * self.hop_length + self.pad
        out = np.zeros(out_len, dtype=np.float32)
        env = np.zeros(out_len, dtype=np.float32)
        out[: self.pad] = self.ola_out
        env[: self.pad] = self.ola_env
        for m in range(C):
            st = m * self.hop_length
            out[st : st + self.n_fft] += time_frames[m]
            env[st : st + self.n_fft] += self.w2

        emit_end = C * self.hop_length
        trim_rem = self.trim_remaining
        if is_final:
            tail_end = emit_end + self.left_trim
            wav = (
                np.zeros(0, dtype=np.float32)
                if tail_end <= trim_rem
                else out[trim_rem:tail_end] / np.maximum(env[trim_rem:tail_end], 1e-6)
            )
            self.reset()
            return wav.astype(np.float32, copy=False)

        if emit_end <= trim_rem:
            wav = np.zeros(0, dtype=np.float32)
            self.trim_remaining = trim_rem - emit_end
        else:
            wav = out[trim_rem:emit_end] / np.maximum(env[trim_rem:emit_end], 1e-6)
            self.trim_remaining = 0
        self.ola_out = out[emit_end : emit_end + self.pad].copy()
        self.ola_env = env[emit_end : emit_end + self.pad].copy()
        return wav.astype(np.float32, copy=False)


class AetherONNXStreamingEngine:
    def __init__(self, onnx_dir: Path, use_int8: bool = False, use_vocoder: bool = True, num_threads: int = 2):
        self.onnx_dir = Path(onnx_dir).expanduser().resolve()
        self.meta = json.loads((self.onnx_dir / "onnx_meta.json").read_text(encoding="utf-8"))
        self.sample_rate = int(self.meta["sample_rate"])
        self.n_fft = int(self.meta["n_fft"])
        self.hop_length = int(self.meta["hop_length"])
        self.n_mels = int(self.meta["n_mels"])
        self.mel_mean = np.asarray(self.meta["mel_mean"], dtype=np.float32).reshape(1, self.n_mels, 1)
        self.ac_cfg = self.meta["streaming"]["acoustic"]
        self.voc_cfg = self.meta["streaming"].get("vocoder") if use_vocoder else None
        self.frontend = BilingualPhonemeFrontend(vocab=self.meta.get("vocab"))

        self.text_sess = _create_session(_resolve_model(self.onnx_dir, "text_step", use_int8), num_threads)
        self.frame_sess = _create_session(_resolve_model(self.onnx_dir, "frame_step", use_int8), num_threads)
        self.voc_sess = (
            _create_session(_resolve_model(self.onnx_dir, "vocoder_step", use_int8), num_threads)
            if self.voc_cfg is not None
            else None
        )

    def _init_states(self):
        ac = self.ac_cfg
        D, nL, H, dh, W = ac["d_model"], ac["n_layers"], ac["n_heads"], ac["head_dim"], ac["window_size"]
        text_st = {"past_len": np.array([0], dtype=np.int64)}
        for i in range(nL):
            text_st[f"k_cache_{i}"] = np.zeros((1, H, W, dh), dtype=np.float32)
            text_st[f"v_cache_{i}"] = np.zeros((1, H, W, dh), dtype=np.float32)
            text_st[f"conv_cache_{i}"] = np.zeros((1, D, ac["conv_pad"]), dtype=np.float32)
        for i, p in enumerate(ac["dur_pads"]):
            text_st[f"dur_cache_{i}"] = np.zeros((1, D, p), dtype=np.float32)

        frame_st = {}
        for prefix, key in (("f0_cache", "f0_pads"), ("energy_cache", "energy_pads"), ("fdec_cache", "fdec_pads")):
            for i, p in enumerate(ac[key]):
                frame_st[f"{prefix}_{i}"] = np.zeros((1, D, p), dtype=np.float32)

        voc_st = None
        if self.voc_cfg is not None:
            vc = self.voc_cfg
            vd = vc["dim"]
            voc_st = {
                "harm_conv1_cache": np.zeros((1, vd, vc["harm_conv1_pad"]), dtype=np.float32),
                "harm_conv2_cache": np.zeros((1, vd // 2, vc["harm_conv2_pad"]), dtype=np.float32),
                "phi": np.zeros((1, 1), dtype=np.float64),
                "prev_f0": np.full((1, 1), -1.0, dtype=np.float64),
            }
            for i, p in enumerate(vc["block_pads"]):
                voc_st[f"block_cache_{i}"] = np.zeros((1, vd, p), dtype=np.float32)
        return text_st, frame_st, voc_st

    def _run_text_step(self, p_win, l_win, b_win, commit_len: int, speed: float, text_st: dict):
        feed = {
            "phoneme_ids": np.asarray([p_win], dtype=np.int64),
            "lang_ids": np.asarray([l_win], dtype=np.int64),
            "boundary_ids": np.asarray([b_win], dtype=np.int64),
            "commit_len": np.asarray([commit_len], dtype=np.int64),
            "speed": np.asarray([speed], dtype=np.float32),
            **text_st,
        }
        outs = self.text_sess.run(None, feed)
        out_names = [o.name for o in self.text_sess.get_outputs()]
        h_frame = outs[0]
        new_st = {}
        for name, val in zip(out_names[2:], outs[2:]):
            new_st[name[4:] if name.startswith("new_") else name] = val
        return h_frame, new_st

    def _run_frame_step(self, h_frame_win: np.ndarray, frame_st: dict):
        outs = self.frame_sess.run(None, {"h_frame_win": h_frame_win, **frame_st})
        out_names = [o.name for o in self.frame_sess.get_outputs()]
        mel = outs[0]
        new_st = {}
        for name, val in zip(out_names[1:], outs[1:]):
            new_st[name[4:] if name.startswith("new_") else name] = val
        return mel, new_st

    def _run_vocoder_step(self, mel_win: np.ndarray, voc_st: dict):
        outs = self.voc_sess.run(None, {"mel_win": mel_win, **voc_st})
        out_names = [o.name for o in self.voc_sess.get_outputs()]
        time_frames = outs[0].squeeze(0)
        new_st = {}
        for name, val in zip(out_names[1:], outs[1:]):
            new_st[name[4:] if name.startswith("new_") else name] = val
        return time_frames, new_st

    def stream_synthesize_ids(self, phoneme_ids: list, lang_ids: list, boundary_ids: list, speed: float = 1.0):
        """流式生成器：逐块产出生效音频分片 (wav_chunk, mel_chunk, is_final)。"""
        ac = self.ac_cfg
        C, L = ac["chunk_size"], ac["lookahead"]
        D = ac["d_model"]
        fp_kl, fp_la = ac["frame_pre_k_left"], ac["frame_pre_lookahead"]
        fp_k = fp_kl + 1 + fp_la

        text_st, frame_st, voc_st = self._init_states()
        h_frame_buf = np.zeros((1, fp_kl, D), dtype=np.float32)

        ola = StreamingOLA(self.n_fft, self.hop_length) if self.voc_sess is not None else None
        if self.voc_sess is not None:
            vc = self.voc_cfg
            v_kl, v_la = vc["k_left"], vc["lookahead"]
            v_keep = v_kl + v_la
            first_m = vc["stream_mel_first_chunk"]
            steady_m = vc["stream_mel_chunk"]
            mel_ctx = np.repeat(self.mel_mean, v_kl, axis=2)  # 初始左侧 k_left 帧填充 mel_mean (归一化为 0)
            mel_pending = np.zeros((1, self.n_mels, 0), dtype=np.float32)
            first_emitted = False

        # 模拟外部流式输入（按 C 个音素分块喂入，保留 L 步前瞻滑窗）
        p_buf, l_buf, b_buf = [], [], []
        T = len(phoneme_ids)
        for pos in range(0, T, C):
            is_last_feed = (pos + C >= T)
            p_buf.extend(phoneme_ids[pos : pos + C])
            l_buf.extend(lang_ids[pos : pos + C])
            b_buf.extend(boundary_ids[pos : pos + C])

            # 1) 文本分块前瞻编码 -> h_frame
            new_h_frames = []
            while len(p_buf) >= C + L:
                hf, text_st = self._run_text_step(p_buf[: C + L], l_buf[: C + L], b_buf[: C + L], C, speed, text_st)
                new_h_frames.append(hf)
                p_buf, l_buf, b_buf = p_buf[C:], l_buf[C:], b_buf[C:]
            if is_last_feed:
                while len(p_buf) > 0:
                    c_len = min(C, len(p_buf))
                    win_end = min(len(p_buf), c_len + L)
                    hf, text_st = self._run_text_step(
                        p_buf[:win_end], l_buf[:win_end], b_buf[:win_end], c_len, speed, text_st
                    )
                    new_h_frames.append(hf)
                    p_buf, l_buf, b_buf = p_buf[c_len:], l_buf[c_len:], b_buf[c_len:]

            if new_h_frames:
                h_frame_buf = np.concatenate([h_frame_buf, *new_h_frames], axis=1)
            if is_last_feed and fp_la > 0:
                h_frame_buf = np.concatenate([h_frame_buf, np.zeros((1, fp_la, D), dtype=np.float32)], axis=1)

            # 2) 帧级韵律与 6 层空洞解码 -> raw log-mel
            mel_chunk = np.zeros((1, self.n_mels, 0), dtype=np.float32)
            if h_frame_buf.shape[1] >= fp_k:
                mel_chunk, frame_st = self._run_frame_step(h_frame_buf, frame_st)
                keep_f = fp_kl + fp_la
                h_frame_buf = h_frame_buf[:, -keep_f:, :] if keep_f > 0 else h_frame_buf[:, :0, :]

            if self.voc_sess is None:
                if mel_chunk.shape[-1] > 0 or is_last_feed:
                    yield np.zeros(0, dtype=np.float32), mel_chunk, is_last_feed
                continue

            # 3) 双档低延迟声码器 + OLA
            if mel_chunk.shape[-1] > 0:
                mel_pending = np.concatenate([mel_pending, mel_chunk], axis=2)

            emitted_mel_sent = False
            while True:
                m_step = first_m if not first_emitted else steady_m
                if mel_pending.shape[-1] < m_step:
                    break
                cur_m = mel_pending[:, :, :m_step]
                mel_pending = mel_pending[:, :, m_step:]
                mel_ctx = np.concatenate([mel_ctx, cur_m], axis=2)
                if mel_ctx.shape[-1] >= v_keep + 1:
                    tf, voc_st = self._run_vocoder_step(mel_ctx, voc_st)
                    mel_ctx = mel_ctx[:, :, -v_keep:] if v_keep > 0 else mel_ctx[:, :, :0]
                    wc = ola.push(tf, is_final=False)
                    if wc.size > 0:
                        first_emitted = True
                        yield wc, (mel_chunk if not emitted_mel_sent else np.zeros((1, self.n_mels, 0), np.float32)), False
                        emitted_mel_sent = True

            if is_last_feed:
                parts = [mel_ctx]
                if mel_pending.shape[-1] > 0:
                    parts.append(mel_pending)
                if v_la > 0:
                    parts.append(np.repeat(self.mel_mean, v_la, axis=2))
                mel_ctx = np.concatenate(parts, axis=2)
                tf = (
                    self._run_vocoder_step(mel_ctx, voc_st)[0]
                    if mel_ctx.shape[-1] >= v_keep + 1
                    else np.zeros((0, self.n_fft), dtype=np.float32)
                )
                wc = ola.push(tf, is_final=True)
                yield wc, (mel_chunk if not emitted_mel_sent else np.zeros((1, self.n_mels, 0), np.float32)), True
            elif mel_chunk.shape[-1] > 0 and not emitted_mel_sent:
                yield np.zeros(0, dtype=np.float32), mel_chunk, False

    def synthesize_ids(self, phoneme_ids: list, lang_ids: list, boundary_ids: list, speed: float = 1.0):
        t0 = time.perf_counter()
        ttfa_ms = None
        wav_chunks, mel_chunks = [], []
        for wc, mc, _ in self.stream_synthesize_ids(phoneme_ids, lang_ids, boundary_ids, speed=speed):
            if mc.shape[-1] > 0:
                mel_chunks.append(mc)
            if wc.size > 0:
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t0) * 1000.0
                wav_chunks.append(wc)
        total_sec = time.perf_counter() - t0
        wav = np.concatenate(wav_chunks) if wav_chunks else np.zeros(0, dtype=np.float32)
        mel = np.concatenate(mel_chunks, axis=2) if mel_chunks else np.zeros((1, self.n_mels, 0), dtype=np.float32)
        audio_sec = max(len(wav) / float(self.sample_rate), 1e-6)
        stats = {
            "n_phonemes": len(phoneme_ids),
            "n_frames": int(mel.shape[-1]),
            "n_chunks": len(wav_chunks),
            "audio_sec": audio_sec,
            "ttfa_ms": ttfa_ms if ttfa_ms is not None else total_sec * 1000.0,
            "total_ms": total_sec * 1000.0,
            "rtf": total_sec / audio_sec,
        }
        return wav, mel, stats

    def synthesize(self, text: str, speed: float = 1.0):
        pids, lids, bids = self.frontend.text_to_ids(text)
        return self.synthesize_ids(pids, lids, bids, speed=speed)


def main():
    ap = argparse.ArgumentParser(description="AetherTTS 轻量纯 CPU ONNX Runtime Python 推理脚本")
    ap.add_argument("--onnx-dir", default="./onnx_export", help="包含 .onnx 与 onnx_meta.json 的目录")
    ap.add_argument("--mode", default="offline", choices=["offline", "streaming"], help="推理模式: offline 或 streaming")
    ap.add_argument("--text", required=True, help="待合成的中/英/混读文本")
    ap.add_argument("--out", default="output.wav", help="输出 WAV 文件路径")
    ap.add_argument("--speed", type=float, default=1.0, help="语速倍率（>1 变快，<1 变慢）")
    ap.add_argument("--threads", type=int, default=2, help="ONNX Runtime CPU 线程数（嵌入式目标默认 2）")
    ap.add_argument("--int8", action="store_true", help="优先加载 *.int8.onnx 量化模型")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if args.mode == "offline":
        engine = AetherONNXOfflineEngine(onnx_dir, use_int8=args.int8, num_threads=args.threads)
        wav, _, _, st = engine.synthesize(args.text, speed=args.speed)
        save_wav(out_path, wav, engine.sample_rate)
        print(
            f"[Offline ONNX] phonemes={st['n_phonemes']} | frames={st['n_frames']} | audio={st['audio_sec']:.2f}s | "
            f"acoustic={st['acoustic_ms']:.1f}ms + vocoder={st['vocoder_ms']:.1f}ms = {st['total_ms']:.1f}ms | "
            f"RTF={st['rtf']:.3f} -> {out_path}"
        )
    else:
        engine = AetherONNXStreamingEngine(onnx_dir, use_int8=args.int8, num_threads=args.threads)
        wav, _, st = engine.synthesize(args.text, speed=args.speed)
        save_wav(out_path, wav, engine.sample_rate)
        print(
            f"[Streaming ONNX] phonemes={st['n_phonemes']} | frames={st['n_frames']} | chunks={st['n_chunks']} | "
            f"audio={st['audio_sec']:.2f}s | TTFA={st['ttfa_ms']:.1f}ms | total={st['total_ms']:.1f}ms | "
            f"RTF={st['rtf']:.3f} -> {out_path}"
        )


if __name__ == "__main__":
    main()
