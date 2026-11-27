#!/usr/bin/env python3
"""
AetherTTS (a9) 网页演示服务 · ONNX Runtime 推理版（推理链路不加载 PyTorch）：
浏览器输入文本 -> 本服务合成 -> 浏览器自动播放（边合成边推流、边收边播）。

推理复用 scripts/infer_onnx.py 的引擎与 scripts/text_frontend.py 的纯 Python 前端，
只依赖 numpy + onnxruntime + pypinyin；模型由 scripts/export_onnx.py 导出。
HTTP 层与网页端由 scripts/tts_http.py 提供，与 PyTorch 版 scripts/aethertts_server.py 的接口、页面完全一致。

网页端可以按请求选择「后端」与「精度」：每个组合在第一次用到时才加载，/api/info 的 options 字段列出当前可选项。
请求里省略 backend / precision 或传 null 表示用服务默认组合。

    后端 backend    streaming（ONNX 三段流式，默认）/ offline（ONNX 离线两段式）/ pytorch（需 --with-pytorch）
    精度 precision  int8 / fp32（以后端实际导出的模型为准）

    GET  /               网页端 (web/index.html)
    GET  /api/info       {"sample_rate", "arch_tag", "device", "backend", "max_chars", "speed_range", "options"}
    POST /api/tts        {"text", "speed", "backend"?, "precision"?} -> 流式返回裸 PCM (audio/pcm; format=s16le; 单声道)
    POST /api/tts/wav    同上 -> 合成完毕后返回完整 WAV (PCM_16)

用法：
    # 1) 先导出模型（默认同时导出离线两段式与三段式流式；--quantize-int8 额外生成权重 INT8 动态量化版）
    conda activate aethertts
    python scripts/export_onnx.py --root ./aethertts_exp --out-dir ./onnx_export --quantize-int8

    # 2) 启动服务（CPU 2 线程；默认后端 streaming，目录里有 INT8 模型时默认精度 INT8）
    python scripts/aethertts_server_onnx.py                                  # http://127.0.0.1:8001
    python scripts/aethertts_server_onnx.py --host 0.0.0.0 --threads 4
    python scripts/aethertts_server_onnx.py --mode offline --precision fp32   # 改默认后端 / 精度（默认 auto：有 INT8 就用 INT8）
    python scripts/aethertts_server_onnx.py --with-pytorch                   # 额外提供 PyTorch 后端用于对比（选中时才加载 torch）

默认只监听 127.0.0.1；--host 0.0.0.0 可供局域网 / WSL 宿主机浏览器访问（无鉴权，请勿直接暴露到公网）。
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tts_http
from scripts.infer_onnx import AetherONNXOfflineEngine, AetherONNXStreamingEngine

MODEL_STEMS = {"streaming": ("text_step", "frame_step", "vocoder_step"), "offline": ("acoustic", "vocoder")}
BACKEND_LABELS = {"streaming": "ONNX 三段流式", "offline": "ONNX 离线两段式", "pytorch": "PyTorch"}
PRECISIONS = ("int8", "fp32")
SUFFIX = {"int8": ".int8.onnx", "fp32": ".onnx"}


class OnnxEngine:
    """单一（后端, 精度）组合的引擎：把 infer_onnx.py 适配成 tts_http.py 的 Engine 协议。"""

    def __init__(self, onnx_dir, mode: str, use_int8: bool, num_threads: int = 2):
        self.mode = mode
        engine_cls = AetherONNXStreamingEngine if mode == "streaming" else AetherONNXOfflineEngine
        self.engine = engine_cls(onnx_dir, use_int8=use_int8, num_threads=num_threads)
        self.sample_rate = self.engine.sample_rate
        meta = self.engine.meta
        voc_cfg = (meta.get("streaming") or {}).get("vocoder") or {}
        # 离线模型整句合成后按流式稳态块长下发，网页端的起播缓冲行为与流式一致
        self.chunk = int(voc_cfg.get("stream_mel_chunk", 16)) * int(meta["hop_length"])

    def _ids(self, text: str):
        pids, lids, bids = self.engine.frontend.text_to_ids(text)
        if len(pids) <= 2:  # 只有 <bos>/<eos>，没有可发音的音素
            raise ValueError("文本中没有可发音的内容")
        return pids, lids, bids

    def stream(self, text: str, speed: float = 1.0):
        """生成器：逐块产出 float32 波形。流式后端由三段式 ONNX 边算边出；无有效音素时在首次迭代抛 ValueError。"""
        ids = self._ids(text)
        if self.mode == "streaming":
            for wav, _, _ in self.engine.stream_synthesize_ids(*ids, speed=speed):
                if wav.size > 0:
                    yield wav
        else:
            wav = self.engine.synthesize_ids(*ids, speed=speed)[0]
            for i in range(0, len(wav), self.chunk):
                yield wav[i : i + self.chunk]

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        ids = self._ids(text)
        if self.mode == "offline":
            return self.engine.synthesize_ids(*ids, speed=speed)[0]
        chunks = [wav for wav, _, _ in self.engine.stream_synthesize_ids(*ids, speed=speed) if wav.size > 0]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def warmup(self):
        self.synthesize("预热。")
        for _ in self.stream("预热。"):
            pass


class SelectableEngine:
    """按请求选择后端与精度：各组合在首次使用时才加载并缓存；对 tts_http.py 暴露 Engine 协议和 select()。"""

    def __init__(self, onnx_dir, num_threads: int = 2, default_backend: str = "streaming", default_precision: str = "auto",
                 with_pytorch: bool = False, pytorch_root: str = "./aethertts_exp", pytorch_device=None):
        self.onnx_dir = Path(onnx_dir).expanduser().resolve()
        self.num_threads, self.pytorch_root, self.pytorch_device = num_threads, pytorch_root, pytorch_device
        meta = json.loads((self.onnx_dir / "onnx_meta.json").read_text(encoding="utf-8"))
        self.sample_rate = int(meta["sample_rate"])

        self.backends = []  # 每项 {"id", "label", "precisions": [...]}，顺序即推荐顺序
        for mode in ("streaming", "offline"):
            precs = [p for p in PRECISIONS if all((self.onnx_dir / f"{stem}{SUFFIX[p]}").exists() for stem in MODEL_STEMS[mode])]
            if precs:
                self.backends.append({"id": mode, "label": BACKEND_LABELS[mode], "precisions": precs})
        if with_pytorch:
            self.backends.append({"id": "pytorch", "label": BACKEND_LABELS["pytorch"], "precisions": ["fp32"]})
        by_id = {b["id"]: b for b in self.backends}
        if not by_id:
            raise FileNotFoundError(
                f"{self.onnx_dir} 下没有任何完整的模型组合: " + " ".join(f"{stem}.onnx" for m in MODEL_STEMS for stem in MODEL_STEMS[m][:1])
                + f"\n       请先运行 python scripts/export_onnx.py --out-dir {self.onnx_dir}"
            )
        if default_backend not in by_id:
            need = ", ".join(f"{stem}{SUFFIX[p]}" for stem in MODEL_STEMS[default_backend] for p in PRECISIONS)
            raise FileNotFoundError(f"{self.onnx_dir} 下没有 {BACKEND_LABELS[default_backend]} 所需的模型: {need}")
        if default_precision != "auto" and default_precision not in by_id[default_backend]["precisions"]:
            need = ", ".join(f"{stem}{SUFFIX[default_precision]}" for stem in MODEL_STEMS[default_backend])
            raise FileNotFoundError(f"{self.onnx_dir} 下没有 {BACKEND_LABELS[default_backend]} 的 {default_precision.upper()} 模型: {need}"
                                   f"\n       请先运行 python scripts/export_onnx.py --out-dir {self.onnx_dir} --quantize-int8，或改用 --precision auto")
        self.default = {"backend": default_backend, "precision": self._auto_precision(by_id[default_backend]) if default_precision == "auto" else default_precision}
        self._cache, self._lock = {}, threading.Lock()
        default_label = self.label(self.default["backend"], self.default["precision"])
        self.info = {
            "arch_tag": "ONNX", "device": "cpu",
            "backend": f"onnxruntime {ort.__version__} · {num_threads} 线程 · 默认 {default_label}",
            "options": {"backends": self.backends, "default": self.default},
        }

    @staticmethod
    def _auto_precision(backend: dict) -> str:
        return "int8" if "int8" in backend["precisions"] else backend["precisions"][0]

    def label(self, backend: str, precision: str) -> str:
        return f"{precision.upper()} · {BACKEND_LABELS[backend]}"

    def select(self, backend: str = None, precision: str = None):
        """返回 (engine, 标签)。缺省项取服务默认值；未知或未导出的组合抛 ValueError。"""
        by_id = {b["id"]: b for b in self.backends}
        backend = backend or self.default["backend"]
        if backend not in by_id:
            raise ValueError(f"未知后端: {backend}（可选: {', '.join(by_id)}）")
        precision = precision or (self.default["precision"] if backend == self.default["backend"] else by_id[backend]["precisions"][0])
        if precision not in by_id[backend]["precisions"]:
            raise ValueError(f"{BACKEND_LABELS[backend]} 不支持精度 {precision}（可选: {', '.join(by_id[backend]['precisions'])}）")
        return self._load(backend, precision), self.label(backend, precision)

    def _load(self, backend: str, precision: str):
        with self._lock:  # 并发的首次请求只加载一次
            if (backend, precision) not in self._cache:
                if backend == "pytorch":
                    try:
                        from scripts.aethertts_infer import AetherTTS  # 只在真的选中 PyTorch 后端时才 import torch
                        engine = AetherTTS(root=self.pytorch_root, device=self.pytorch_device)
                    except (ImportError, FileNotFoundError, RuntimeError) as e:
                        raise RuntimeError(f"加载 PyTorch 后端失败（--pytorch-root {self.pytorch_root}）: {e}") from e
                else:
                    engine = OnnxEngine(self.onnx_dir, mode=backend, use_int8=(precision == "int8"), num_threads=self.num_threads)
                engine.warmup()
                self._cache[(backend, precision)] = engine
            return self._cache[(backend, precision)]

    def warmup(self):
        self._load(self.default["backend"], self.default["precision"])


def main():
    ap = argparse.ArgumentParser(description="AetherTTS (a9) web demo server (ONNX Runtime, no PyTorch): text -> streaming speech.")
    tts_http.add_common_args(ap, port=8001)
    ap.add_argument("--onnx-dir", default="./onnx_export", help="export_onnx.py 的输出目录 (含 .onnx 与 onnx_meta.json).")
    ap.add_argument("--mode", default="streaming", choices=list(MODEL_STEMS), help="默认后端: streaming 三段式流式 (默认) / offline 离线两段式.")
    ap.add_argument("--threads", type=int, default=2, help="ONNX Runtime CPU 线程数 (默认 2，与嵌入式目标一致).")
    ap.add_argument("--precision", default="auto", choices=["auto", "int8", "fp32"],
                    help="默认精度: auto 有 INT8 模型就用 INT8 否则 FP32 (默认); int8 / fp32 强制指定，缺文件直接报错.")
    ap.add_argument("--with-pytorch", action="store_true", help="额外提供 PyTorch 后端用于对比（选中时才加载 torch）.")
    ap.add_argument("--pytorch-root", default="./aethertts_exp", help="PyTorch 后端读取的 checkpoint 目录 (runs / vocoder).")
    ap.add_argument("--pytorch-device", default=None, help="PyTorch 后端的设备: cuda / cpu (默认有 CUDA 则用 CUDA).")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    if not (onnx_dir / "onnx_meta.json").exists():
        sys.exit(f"[Error] {onnx_dir} 下没有 onnx_meta.json：请先运行 python scripts/export_onnx.py --out-dir {args.onnx_dir}")
    try:
        engine = SelectableEngine(
            onnx_dir, num_threads=args.threads, default_backend=args.mode, default_precision=args.precision,
            with_pytorch=args.with_pytorch, pytorch_root=args.pytorch_root, pytorch_device=args.pytorch_device,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[Error] {e}")
    engine.warmup()
    combos = " / ".join(f"{engine.label(b['id'], p)}" for b in engine.backends for p in b["precisions"])
    print(
        f"[Load] onnxruntime {ort.__version__} | device=cpu | {engine.sample_rate} Hz | onnx_dir: {onnx_dir}\n"
        f"       可选: {combos}\n"
        f"       PyTorch {'已加载' if 'torch' in sys.modules else '未加载（纯 onnxruntime + numpy 推理）'}",
        flush=True,
    )
    tts_http.serve(engine, host=args.host, port=args.port, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
