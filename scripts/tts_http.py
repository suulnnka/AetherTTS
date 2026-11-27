#!/usr/bin/env python3
"""
AetherTTS 网页服务的通用 HTTP 层（仅标准库 + numpy）：网页端 + 流式 PCM + 完整 WAV 接口。

推理引擎由调用方注入，只需满足 Engine 协议，PyTorch 与 ONNX Runtime 两个服务共用本模块：

    engine.sample_rate : int
    engine.info        : dict            /api/info 的引擎相关字段（arch_tag / device / backend / options ...）
    engine.stream(text, speed) -> 可迭代的 float32 波形块 (np.ndarray)；无有效音素时在首次迭代抛 ValueError
    engine.synthesize(text, speed) -> float32 波形 (np.ndarray)
    engine.warmup()                      提前完成词典加载 / 设备预热
    engine.select(backend, precision)    可选：支持按请求选择后端/精度的引擎实现它，返回 (该组合的 engine, 日志标签)

    GET  /               网页端 (web/index.html)
    GET  /api/info       {"sample_rate", "arch_tag", "device", "backend", "max_chars", "speed_range", "options"?}
    POST /api/tts        {"text", "speed", "backend"?, "precision"?} -> 流式返回裸 PCM (audio/pcm; format=s16le; 单声道)，每个音频块立即写出
    POST /api/tts/wav    同上 -> 合成完毕后返回完整 WAV (PCM_16)

用法见 scripts/aethertts_server.py（PyTorch）与 scripts/aethertts_server_onnx.py（ONNX Runtime）。
"""
import io
import itertools
import json
import signal
import socket
import struct
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_BODY_BYTES = 64 * 1024
SPEED_RANGE = (0.5, 2.0)
DEFAULT_MAX_CHARS = 500


def pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class TTSServer(ThreadingHTTPServer):
    """每个请求一个线程；推理按「单个音频块」粒度加锁串行（前端词典 / 推理会话非线程安全），网络读写不占锁。"""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, addr, handler, engine, index_html: bytes, max_chars: int):
        super().__init__(addr, handler)
        self.engine, self.index_html, self.max_chars = engine, index_html, max_chars
        self._infer_lock = threading.Lock()
        self.info = {
            "sample_rate": engine.sample_rate, "max_chars": max_chars, "speed_range": list(SPEED_RANGE),
            **getattr(engine, "info", {}),
        }

    def locked_iter(self, gen):
        it = iter(gen)
        while True:
            with self._infer_lock:
                try:
                    chunk = next(it)
                except StopIteration:
                    return
            yield chunk

    def locked_call(self, fn, *args):
        with self._infer_lock:
            return fn(*args)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AetherTTS"
    disable_nagle_algorithm = True  # 小块音频立即发出，不被 Nagle 攒包
    timeout = 60

    def log_message(self, fmt, *args):  # 默认实现写 stderr 缓冲不实时刷新，这里改为实时
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    def log_request(self, code="-", size="-"):
        pass  # 各路由自行打印单行摘要，避免与默认访问日志重复

    # ---- 基础收发 ----
    def _send(self, status: int, body: bytes, ctype: str, extra: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict):
        if status >= 400:
            self.log_message("%s %d %s", self.requestline, status, obj.get("error", ""))
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_request(self):
        """返回 (text, speed, sel) 或在出错时已回送 4xx 并返回 None；sel 是请求里的 backend / precision 选择（可为空 dict）。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "请求体过大"})
            return None
        raw = self.rfile.read(length) if length else b""
        sel = {}
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("请求体必须是 JSON 对象")
            text = body.get("text", "")
            if not isinstance(text, str):
                raise ValueError("text 必须是字符串")
            speed = float(body.get("speed", 1.0))
            if not (SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]):
                raise ValueError(f"speed 需在 {SPEED_RANGE[0]} ~ {SPEED_RANGE[1]} 之间")
            for key in ("backend", "precision"):
                val = body.get(key)
                if val is not None:
                    if not isinstance(val, str) or not val:
                        raise ValueError(f"{key} 必须是非空字符串")
                    sel[key] = val
        except (ValueError, TypeError) as e:
            self._json(400, {"error": f"请求参数无效: {e}"})
            return None
        text = text.strip()
        if not text:
            self._json(400, {"error": "text 不能为空"})
            return None
        if len(text) > self.server.max_chars:
            self._json(400, {"error": f"文本过长：{len(text)} 字（上限 {self.server.max_chars} 字）"})
            return None
        return text, speed, sel

    def _pick_engine(self, sel: dict):
        """返回本次请求使用的 (engine, 日志标签)。带 backend / precision 的请求只有引擎支持选择时才有效。"""
        select = getattr(self.server.engine, "select", None)
        if select is None:  # 单一引擎的服务
            if sel:
                raise ValueError("该服务不支持选择后端 / 精度")
            return self.server.engine, ""
        return select(**sel)  # 未指定的项取服务默认组合

    # ---- 路由 ----
    def do_GET(self):
        srv = self.server
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, srv.index_html, "text/html; charset=utf-8")
                self.log_message("GET / 200 网页端")
            elif self.path == "/api/info":
                self._json(200, srv.info)
            else:
                self._json(404, {"error": "Not Found"})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        try:
            if self.path == "/api/tts":
                self._tts_stream()
            elif self.path == "/api/tts/wav":
                self._tts_wav()
            else:
                self._json(404, {"error": "Not Found"})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _tts_wav(self):
        req = self._read_request()
        if req is None:
            return
        text, speed, sel = req
        srv = self.server
        try:
            eng, tag = self._pick_engine(sel)
            t0 = time.perf_counter()
            wav = srv.locked_call(eng.synthesize, text, speed)
        except ValueError as e:  # 无效的后端 / 精度组合，或引擎判定无有效音素
            self._json(400, {"error": str(e)})
            return
        except (RuntimeError, FileNotFoundError) as e:  # 例如所选后端的权重文件缺失
            self._json(500, {"error": str(e)})
            return
        self._send(200, wav_bytes(pcm16(wav), eng.sample_rate), "audio/wav")
        self.log_message('POST /api/tts/wav 200%s "%s" | %.2fs 音频 | 合成 %.0f ms',
                         f" [{tag}]" if tag else "", text[:30], len(wav) / eng.sample_rate, (time.perf_counter() - t0) * 1000)

    def _tts_stream(self):
        req = self._read_request()
        if req is None:
            return
        text, speed, sel = req
        srv = self.server
        try:
            eng, tag = self._pick_engine(sel)
        except ValueError as e:  # 无效的后端 / 精度组合
            self._json(400, {"error": str(e)})
            return
        except (RuntimeError, FileNotFoundError) as e:  # 例如所选后端的权重文件缺失
            self._json(500, {"error": str(e)})
            return
        sr = eng.sample_rate
        gen = srv.locked_iter(eng.stream(text, speed))
        t0 = time.perf_counter()
        try:
            first = next(gen, None)  # 先算出首块再回响应头：引擎报错（如空文本）还能返回 400
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header("Content-Type", f"audio/pcm; format=s16le; rate={sr}; channels=1")
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        first_ms = (time.perf_counter() - t0) * 1000
        n_samples, n_chunks = 0, 0
        try:
            for chunk in itertools.chain(() if first is None else (first,), gen):
                data = pcm16(chunk)
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
                self.wfile.flush()
                n_samples += len(chunk)
                n_chunks += 1
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.log_message('POST /api/tts 200%s "%s" | 客户端中途断开（已发 %d 块）', f" [{tag}]" if tag else "", text[:30], n_chunks)
            self.close_connection = True
            return
        except Exception as e:  # 响应头已发出，只能中断连接让客户端报网络错误
            self.log_message('POST /api/tts 500%s "%s" | 合成中断: %s: %s', f" [{tag}]" if tag else "", text[:30], type(e).__name__, e)
            self.close_connection = True
            return
        self.log_message(
            'POST /api/tts 200%s "%s" | %.2fs 音频 / %d 块 | 首块 %.0f ms | 总 %.0f ms',
            f" [{tag}]" if tag else "", text[:30], n_samples / sr, n_chunks, first_ms, (time.perf_counter() - t0) * 1000,
        )


def _default_gateway() -> str:
    """默认路由网关地址（Linux 读 /proc/net/route）；取不到时退回公网地址，UDP connect 不会真正发包。"""
    try:
        with open("/proc/net/route") as f:
            for line in f.read().splitlines()[1:]:
                fields = line.split()
                if fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, ValueError, IndexError):
        pass
    return "8.8.8.8"


def lan_ip() -> str:
    """本机在默认路由出口上的地址（WSL 下取到的是 eth0 的局域网 IP，而不是 DNS 隧道的回环别名）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((_default_gateway(), 9))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _exit_on_sigterm(signum, frame):
    raise KeyboardInterrupt  # 走与 Ctrl+C 相同的收尾逻辑


def add_common_args(ap, port: int = 8000):
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机；0.0.0.0 供局域网 / WSL 宿主机访问）.")
    ap.add_argument("--port", type=int, default=port)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="单次请求文本字数上限.")


def serve(engine, host: str, port: int, max_chars: int = DEFAULT_MAX_CHARS):
    """阻塞运行网页服务。engine 需满足模块文档中的 Engine 协议，且已完成 warmup()。"""
    # aethertts_lib 在 import 时会把 SIGTERM 换成只置标志位的训练用处理器；服务进程需要 kill / systemd 能正常结束
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    index_html = (REPO_ROOT / "web" / "index.html").read_bytes()
    try:
        httpd = TTSServer((host, port), Handler, engine, index_html, max_chars)
    except OSError as e:
        sys.exit(f"[Error] 无法监听 {host}:{port}: {e.strerror}（端口被占用？换一个 --port）")
    shown = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    print(f"[Serve] 网页端: http://{shown}:{port}", flush=True)
    if host in ("0.0.0.0", "::"):
        print(f"[Serve] 局域网: http://{lan_ip()}:{port}  (无鉴权，请勿暴露到公网)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Stop] 服务已关闭（Ctrl+C 或 SIGTERM）", flush=True)
    finally:
        httpd.server_close()
