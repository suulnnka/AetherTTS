#!/usr/bin/env python3
"""
AetherTTS (a9) 网页演示服务 · PyTorch 推理版：浏览器输入文本 -> 本服务合成 -> 浏览器自动播放（边合成边推流、边收边播）。

推理走 scripts/aethertts_infer.py 的 AetherTTS（PyTorch 权重）；HTTP 层与网页端由 scripts/tts_http.py 提供，
与 ONNX Runtime 版 scripts/aethertts_server_onnx.py 共用同一套接口和页面。

    GET  /               网页端 (web/index.html)
    GET  /api/info       {"sample_rate", "arch_tag", "device", "backend", "max_chars", "speed_range"}
    POST /api/tts        {"text", "speed"} -> 流式返回裸 PCM (audio/pcm; format=s16le; 单声道)，浏览器用 Web Audio 边收边播
    POST /api/tts/wav    {"text", "speed"} -> 合成完毕后返回完整 WAV (PCM_16)

用法：
    conda activate aethertts
    python scripts/aethertts_server.py                                  # http://127.0.0.1:8000
    python scripts/aethertts_server.py --host 0.0.0.0 --port 8000 --device cpu
    curl -s -X POST localhost:8000/api/tts/wav -H 'Content-Type: application/json' \\
         -d '{"text":"你好，请问今天北京的天气怎么样？"}' --output hello.wav
    curl -sN -X POST localhost:8000/api/tts -H 'Content-Type: application/json' \\
         -d '{"text":"你好"}' | ffplay -f s16le -ar 24000 -ac 1 -autoexit -

默认只监听 127.0.0.1；--host 0.0.0.0 可供局域网 / WSL 宿主机浏览器访问（无鉴权，请勿直接暴露到公网）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts import tts_http
from scripts.aethertts_infer import AetherTTS


def main():
    ap = argparse.ArgumentParser(description="AetherTTS (a9) web demo server (PyTorch): text -> streaming speech.")
    tts_http.add_common_args(ap)
    ap.add_argument("--root", default="./aethertts_exp", help="训练产物目录 (runs / vocoder).")
    ap.add_argument("--device", default=None, help="cuda / cpu (默认有 CUDA 则用 CUDA).")
    ap.add_argument("--acoustic-ckpt", default=None, help="声学 checkpoint (默认自动发现最新 best.pt).")
    ap.add_argument("--vocoder-ckpt", default=None, help="声码器 checkpoint (默认自动匹配配套 final.pt).")
    args = ap.parse_args()

    tts = AetherTTS(root=args.root, device=args.device, acoustic_ckpt=args.acoustic_ckpt, vocoder_ckpt=args.vocoder_ckpt)
    tts.warmup()  # 首个请求不再承担词典加载 / 设备内核预热
    tts.info = {"arch_tag": tts.arch_tag, "device": str(tts.device), "backend": f"PyTorch {torch.__version__}"}
    print(
        f"[Load] arch={tts.arch_tag} | backend=PyTorch {torch.__version__} | device={tts.device} | {tts.sample_rate} Hz\n"
        f"       acoustic: {tts.acoustic_path}\n       vocoder : {tts.vocoder_path}",
        flush=True,
    )
    tts_http.serve(tts, host=args.host, port=args.port, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
