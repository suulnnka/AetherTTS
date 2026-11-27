#!/usr/bin/env python3
"""
AetherTTS Phase 1 (a9) ONNX 导出与数值一致性校验脚本。

支持将训练好的声学模型 (AetherAcousticModel, 默认 E3_chunk_attn_C4_L8) 与
连续相位谐波源因果声码器 (CausalVocosLite) 导出为纯标准 ONNX 算子图（零自定义算子、
零复数张量、零 F.fold / aten::unfold / torch.searchsorted 导出陷阱）：

  1. 离线两段式 (--mode offline):
     - acoustic.onnx : (phoneme_ids, lang_ids, boundary_ids, speed) -> (mel, durations)
     - vocoder.onnx  : (mel) -> (wav)
       * 内置 32x32 Cooley-Tukey 纯实数 rfft/irfft 与 4 路移位相加 OLA，体积仅增 ~24 KB

  2. 三段式流式 (--mode streaming，对应 Phase 3 路线图):
     - text_step.onnx    : 固定尺寸右对齐 KV Cache 的 ChunkLookaheadBackbone + Duration + 帧相位扩展
     - frame_step.onnx   : 共享 frame_pre + F0/UV/Energy + 6 层空洞帧解码 + Mel 反归一化
     - vocoder_step.onnx : Lookahead 嵌入 + float64 连续相位谐波源 + 8 层空洞 ConvNeXt + 窗函数时域帧

  3. 全量模式 (--mode both，默认): 同时导出离线 + 三段流式 ONNX，并生成 onnx_meta.json。

用法示例:
    # 自动从 ./aethertts_exp 发现最新 best.pt 与声码器 final.pt 并导出 + 校验
    python scripts/export_onnx.py --root ./aethertts_exp --out-dir ./onnx_export

    # 手动指定权重路径（仅导出离线两段式）
    python scripts/export_onnx.py --mode offline \\
        --acoustic-ckpt ./aethertts_exp/runs/E3_.../best.pt \\
        --vocoder-ckpt  ./aethertts_exp/vocoder/causal_vocos_lite_.../final.pt \\
        --out-dir ./onnx_export

    # 无训练权重时用随机权重做导出与数值等价门禁自检
    python scripts/export_onnx.py --dummy --out-dir ./onnx_export_smoke
"""
import argparse
import inspect
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import aethertts_lib as lib
from scripts.text_frontend import BilingualPhonemeFrontend as PureTextFrontend


# ==============================================================================
# 1. Checkpoint 自动发现与加载
# ==============================================================================
def find_acoustic_ckpt(root: Path, model_name: str, profile: str, dataset: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"指定的声学 checkpoint 不存在: {p}")
        return p
    runs_dir = root / "runs"
    cands = []
    if runs_dir.exists():
        for pat in (
            f"{model_name}_{dataset}_{profile}_a9*",
            f"{model_name}_{dataset}_{profile}_*",
            f"{model_name}_{dataset}_*",
            f"{model_name}*",
        ):
            cands = sorted(runs_dir.glob(pat))
            if cands:
                break
    for d in reversed(cands):
        for fn in ("best.pt", "latest.pt"):
            if (d / fn).exists():
                return d / fn
    raise FileNotFoundError(
        f"在 {runs_dir} 中未找到 {model_name} 的 best.pt / latest.pt。"
        "请通过 --acoustic-ckpt 指定路径，或使用 --dummy 进行随机权重自检导出。"
    )


def find_vocoder_ckpt(root: Path, profile: str, dataset: str, explicit: str = ""):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"指定的声码器 checkpoint 不存在: {p}")
        return p
    voc_dir = root / "vocoder"
    if not voc_dir.exists():
        return None
    cands = []
    for pat in (
        f"causal_vocos_lite_{dataset}_{profile}_a9*",
        f"causal_vocos_lite_{dataset}_{profile}_*",
        f"causal_vocos_lite_{dataset}_*",
        "causal_vocos_lite_*",
    ):
        cands = sorted(voc_dir.glob(pat))
        if cands:
            break
    for d in reversed(cands):
        for fn in ("final.pt", "latest.pt"):
            if (d / fn).exists():
                return d / fn
    return None


def load_acoustic_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    stats = ckpt.get("stats", None)
    vocab = ckpt.get("vocab", lib.FRONTEND.vocab)
    bb_type = ckpt.get("backbone_type", "chunk_attn")
    lookahead = int(ckpt.get("lookahead", 8))

    model = lib.AetherAcousticModel(
        vocab_size=len(vocab),
        backbone_type=bb_type,
        d_model=int(cfg.get("d_model", 256)),
        n_mels=int(cfg.get("n_mels", 100)),
        k_left=int(cfg.get("k_left", 3)),
        lookahead=lookahead,
        swa_window=int(cfg.get("swa_window", 48)),
        chunk_size=int(cfg.get("chunk_attn_size", 4)),
        frame_lookahead=int(cfg.get("frame_lookahead", 2)),
        dropout=float(cfg.get("dropout_acoustic", 0.0)),
    )
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    # 剥离仅训练期使用的 aligner 与中层深监督头
    model = model.export_inference_copy().float().eval()
    return model, cfg, stats, list(vocab), ckpt


def load_vocoder_model(voc_ckpt_path: Path, cfg: dict):
    raw = torch.load(voc_ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["voc"] if isinstance(raw, dict) and "voc" in raw else raw

    # 直接从权重形状自动推断 dim / lookahead / n_layers / n_mels，防止 cfg 缺失或不匹配
    n_mels = int(sd["mel_mean"].shape[1]) if "mel_mean" in sd else int(cfg.get("n_mels", 100))
    dim = int(sd["norm_in.weight"].shape[0]) if "norm_in.weight" in sd else int(cfg.get("vocoder_dim", 320))
    n_bins = int(sd["head.out.bias"].shape[0]) // 4 if "head.out.bias" in sd else (int(cfg.get("n_fft", 1024)) // 2 + 1)
    n_fft = (n_bins - 1) * 2
    kernel_size = int(sd["blocks.0.dwconv.conv.weight"].shape[-1]) if "blocks.0.dwconv.conv.weight" in sd else 7
    embed_k = int(sd["embed.conv.weight"].shape[-1]) if "embed.conv.weight" in sd else (kernel_size + int(cfg.get("vocoder_lookahead", 4)))
    lookahead = embed_k - kernel_size
    n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".norm.weight")) or 8

    voc = lib.CausalVocosLite(
        n_mels=n_mels,
        dim=dim,
        n_layers=n_layers,
        kernel_size=kernel_size,
        n_fft=n_fft,
        hop_length=int(cfg.get("hop_length", 256)),
        lookahead=lookahead,
        sample_rate=int(cfg.get("sample_rate", 24000)),
    )
    voc.load_state_dict(sd, strict=True)
    return voc.float().eval()


# ==============================================================================
# 2. ONNX 友好算子组件（Cooley-Tukey 2D FFT / RoPE / Duration 帧扩展）
# ==============================================================================
def _apply_rope_onnx(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class FrameExpanderONNX(nn.Module):
    """无 torch.searchsorted / repeat_interleave 的纯标准 ONNX 帧扩展与音素内相位嵌入（B=1）。"""

    def __init__(self, frame_phase_proj: nn.Linear):
        super().__init__()
        self.frame_phase_proj = frame_phase_proj
        self.register_buffer(
            "prog_mults",
            torch.tensor([2.0 * math.pi * k for k in (1, 2, 4, 8)], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "inv_scales",
            torch.tensor([1.0 / 2.0, 1.0 / 6.0, 1.0 / 18.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, h_txt: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        # h_txt: [1, T, D], durations: [1, T] (int64, >= 1)
        ends = torch.cumsum(durations, dim=1)       # [1, T]
        starts = ends - durations                   # [1, T]
        total_frames = ends[0, -1]                  # scalar int64
        idx = torch.arange(total_frames, device=durations.device, dtype=torch.long)  # [F]
        # ends 单调递增且 idx < ends[-1]，(idx >= ends) 求和严格等价于 searchsorted(right=True)
        tok_idx = (idx.unsqueeze(1) >= ends[0].unsqueeze(0)).to(torch.long).sum(dim=1)  # [F]

        h_exp = h_txt.index_select(1, tok_idx)      # [1, F, D]
        s_f = starts[0].index_select(0, tok_idx)    # [F]
        d_f = durations[0].index_select(0, tok_idx)  # [F]

        tau_f = (idx - s_f).to(torch.float32)
        dur_ff = d_f.clamp(min=1).to(torch.float32)
        rem_f = (dur_ff - 1.0 - tau_f).clamp(min=0.0)
        prog = (tau_f + 0.5) / dur_ff

        prog_angles = prog.unsqueeze(-1) * self.prog_mults
        tau_angles = tau_f.unsqueeze(-1) * self.inv_scales
        rem_angles = rem_f.unsqueeze(-1) * self.inv_scales
        scalars = torch.stack(
            [
                (tau_f / 32.0).clamp(max=2.0),
                (rem_f / 32.0).clamp(max=2.0),
                prog,
                torch.log1p(dur_ff) / 4.0,
            ],
            dim=-1,
        )
        feats = torch.cat(
            [
                torch.sin(prog_angles),
                torch.cos(prog_angles),
                torch.sin(tau_angles),
                torch.cos(tau_angles),
                torch.sin(rem_angles),
                torch.cos(rem_angles),
                scalars,
            ],
            dim=-1,
        ).unsqueeze(0)  # [1, F, 24]
        return h_exp + self.frame_phase_proj(feats).to(h_exp.dtype)


class CooleyTukeyFFT1D(nn.Module):
    """纯实数 MatMul 实现的定长 1D rfft / irfft（N = n1 * n2，1024 = 32 x 32）。

    优势：
    - 零 complex64 张量、零 ONNX DFT 版本差异，兼容 opset 14+ 与全部端侧引擎 (ORT/MNN/NCNN)；
    - 全部预计算矩阵仅占 ~24 KB（较直接 1024x513 稠密 DFT 矩阵省 350x 体积、省 16x 算力）；
    - 与 torch.fft.rfft / torch.fft.irfft 数值误差 ~1e-6。
    """

    def __init__(self, n_fft: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        n1 = int(math.isqrt(n_fft))
        while n1 > 1 and n_fft % n1 != 0:
            n1 -= 1
        n2 = n_fft // n1
        self.n1, self.n2 = n1, n2

        k1 = torch.arange(n1, dtype=torch.float64)
        k2 = torch.arange(n2, dtype=torch.float64)
        a1 = -2.0 * math.pi * (k1[:, None] * k1[None, :]) / float(n1)
        atw = -2.0 * math.pi * (k2[:, None] * k1[None, :]) / float(n_fft)
        a2 = -2.0 * math.pi * (k2[:, None] * k2[None, :]) / float(n2)

        self.register_buffer("w1_re", torch.cos(a1).float(), persistent=False)
        self.register_buffer("w1_im", torch.sin(a1).float(), persistent=False)
        self.register_buffer("tw_re", torch.cos(atw).float(), persistent=False)
        self.register_buffer("tw_im", torch.sin(atw).float(), persistent=False)
        self.register_buffer("w2_re", torch.cos(a2).float(), persistent=False)
        self.register_buffer("w2_im", torch.sin(a2).float(), persistent=False)

        self.register_buffer("iw1_re", torch.cos(a1).float(), persistent=False)
        self.register_buffer("iw1_im", (-torch.sin(a1)).float(), persistent=False)
        self.register_buffer("itw_re", torch.cos(atw).float(), persistent=False)
        self.register_buffer("itw_im", (-torch.sin(atw)).float(), persistent=False)
        self.register_buffer("iw2_re", (torch.cos(a2) / float(n_fft)).float(), persistent=False)
        self.register_buffer("iw2_im", (-torch.sin(a2) / float(n_fft)).float(), persistent=False)

        self.register_buffer("mirror_idx", torch.arange(n_fft // 2 - 1, 0, -1, dtype=torch.long), persistent=False)

    def rfft(self, x: torch.Tensor):
        # x: [B, F, n_fft] (real) -> (re, im) each [B, F, n_bins]
        B, F_len, _ = x.shape
        x_t = x.reshape(B, F_len, self.n1, self.n2).transpose(-1, -2)  # [B, F, n2, n1]
        a_re = torch.matmul(x_t, self.w1_re)
        a_im = torch.matmul(x_t, self.w1_im)
        b_re = a_re * self.tw_re - a_im * self.tw_im
        b_im = a_re * self.tw_im + a_im * self.tw_re
        b_re_t = b_re.transpose(-1, -2)
        b_im_t = b_im.transpose(-1, -2)
        c_re = torch.matmul(b_re_t, self.w2_re) - torch.matmul(b_im_t, self.w2_im)
        c_im = torch.matmul(b_re_t, self.w2_im) + torch.matmul(b_im_t, self.w2_re)
        out_re = c_re.transpose(-1, -2).reshape(B, F_len, self.n_fft)[..., : self.n_bins]
        out_im = c_im.transpose(-1, -2).reshape(B, F_len, self.n_fft)[..., : self.n_bins]
        return out_re, out_im

    def irfft(self, s_re: torch.Tensor, s_im: torch.Tensor) -> torch.Tensor:
        # s_re, s_im: [B, F, n_bins] -> real [B, F, n_fft]
        B, F_len, _ = s_re.shape
        full_re = torch.cat([s_re, s_re.index_select(-1, self.mirror_idx)], dim=-1)
        full_im = torch.cat([s_im, -s_im.index_select(-1, self.mirror_idx)], dim=-1)
        x_re_t = full_re.reshape(B, F_len, self.n1, self.n2).transpose(-1, -2)
        x_im_t = full_im.reshape(B, F_len, self.n1, self.n2).transpose(-1, -2)
        a_re = torch.matmul(x_re_t, self.iw1_re) - torch.matmul(x_im_t, self.iw1_im)
        a_im = torch.matmul(x_re_t, self.iw1_im) + torch.matmul(x_im_t, self.iw1_re)
        b_re = a_re * self.itw_re - a_im * self.itw_im
        b_im = a_re * self.itw_im + a_im * self.itw_re
        b_re_t = b_re.transpose(-1, -2)
        b_im_t = b_im.transpose(-1, -2)
        c_re = torch.matmul(b_re_t, self.iw2_re) - torch.matmul(b_im_t, self.iw2_im)
        return c_re.transpose(-1, -2).reshape(B, F_len, self.n_fft)


# ==============================================================================
# 3. 离线两段式 ONNX 包装器：acoustic.onnx + vocoder.onnx
# ==============================================================================
class ChunkBackboneOfflineONNX(nn.Module):
    """用标准 Gather + 显式 Attention 替代 Tensor.unfold 与 SDPA 的 ChunkLookaheadBackbone。"""

    def __init__(self, backbone: lib.ChunkLookaheadBackbone):
        super().__init__()
        self.backbone = backbone
        self.chunk_size = backbone.chunk_size
        self.lookahead = backbone.lookahead
        self.window_size = backbone.layers[0].window_size
        self.d_model = backbone.d_model
        M = self.chunk_size + self.lookahead
        W = self.window_size
        layer0 = backbone.layers[0]
        # 静态缓存局部窗口 [0 .. W+M) 的 RoPE 表，避免运行时动态重算分支
        cos, sin = layer0._get_local_rope(W + M, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos.clone(), persistent=False)
        self.register_buffer("rope_sin", sin.clone(), persistent=False)
        self.register_buffer("win_offs", torch.arange(M, dtype=torch.long).unsqueeze(0), persistent=False)
        if W > 0:
            self.register_buffer("past_offs", torch.arange(W, dtype=torch.long).unsqueeze(0), persistent=False)
        pad_left = layer0.conv.dwconv.pad_left
        if pad_left > 0:
            self.register_buffer("conv_offs", torch.arange(pad_left, dtype=torch.long).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, T, D]
        B, T, D = x.shape
        C, L, W = self.chunk_size, self.lookahead, self.window_size
        M = C + L
        N = (T + C - 1) // C
        T_pad = (N - 1) * C + M
        pad_r = T_pad - T
        x_pad = torch.cat([x, torch.zeros((B, pad_r, D), dtype=x.dtype, device=x.device)], dim=1)

        starts = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(1) * C  # [N, 1]
        idx_win = (starts + self.win_offs).reshape(-1)                                # [N * M]
        x_win = x_pad.index_select(1, idx_win).reshape(B, N, M, D)
        win_mask = (idx_win < T).reshape(1, N, M)

        if W > 0:
            past_pos = starts - W + self.past_offs                                    # [N, W]
            past_valid = (past_pos >= 0).unsqueeze(0)                                 # [1, N, W]
            idx_past = (starts + self.past_offs).reshape(-1)                          # [N * W]
            key_valid = torch.cat([past_valid, win_mask], dim=2)                      # [1, N, W + M]
        else:
            idx_past = None
            key_valid = win_mask

        h = x_win
        last_idx = len(self.backbone.layers) - 1
        cos_k = self.rope_cos[None, None, None, : W + M, :]
        sin_k = self.rope_sin[None, None, None, : W + M, :]

        for idx, layer in enumerate(self.backbone.layers):
            H, dh = layer.n_heads, layer.head_dim
            q_len = C if (idx == last_idx and C < M) else M
            xn = layer.norm1(h)
            qkv = layer.qkv(xn).reshape(B, N, M, 3, H, dh)
            q = qkv[:, :, :q_len, 0].permute(0, 1, 3, 2, 4)
            k_raw = qkv[:, :, :, 1].permute(0, 1, 3, 2, 4)
            v = qkv[:, :, :, 2].permute(0, 1, 3, 2, 4)

            if W > 0:
                k_commit = k_raw[:, :, :, :C, :].permute(0, 2, 1, 3, 4).reshape(B, H, N * C, dh)
                v_commit = v[:, :, :, :C, :].permute(0, 2, 1, 3, 4).reshape(B, H, N * C, dh)
                zeros_kv = torch.zeros((B, H, W, dh), dtype=k_commit.dtype, device=x.device)
                k_padded = torch.cat([zeros_kv, k_commit], dim=2)
                v_padded = torch.cat([zeros_kv, v_commit], dim=2)
                k_past = k_padded.index_select(2, idx_past).reshape(B, H, N, W, dh).transpose(1, 2)
                v_past = v_padded.index_select(2, idx_past).reshape(B, H, N, W, dh).transpose(1, 2)
                k_cat_raw = torch.cat([k_past, k_raw], dim=3)
                v_all = torch.cat([v_past, v], dim=3)
            else:
                k_cat_raw, v_all = k_raw, v

            cos_q = self.rope_cos[None, None, None, W : W + q_len, :]
            sin_q = self.rope_sin[None, None, None, W : W + q_len, :]
            q_rot = _apply_rope_onnx(q, cos_q, sin_q)
            k_all = _apply_rope_onnx(k_cat_raw, cos_k, sin_k)

            scores = torch.matmul(q_rot, k_all.transpose(-2, -1)) * (1.0 / math.sqrt(float(dh)))
            scores = torch.where(key_valid.unsqueeze(2).unsqueeze(3), scores, torch.full_like(scores, -1e9))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_all).transpose(2, 3).reshape(B, N, q_len, D)
            h_res = h[:, :, :q_len, :] + layer.out_proj(out)

            pad_left = layer.conv.dwconv.pad_left
            if pad_left > 0:
                h_commit = h_res[:, :, :C, :].reshape(B, N * C, D)
                h_padded = torch.cat([torch.zeros((B, pad_left, D), dtype=h_res.dtype, device=x.device), h_commit], dim=1)
                idx_conv = (starts + self.conv_offs).reshape(-1)
                conv_state = h_padded.index_select(1, idx_conv).reshape(B, N, pad_left, D).transpose(2, 3)
                buf = torch.cat([conv_state, h_res.transpose(2, 3)], dim=3).reshape(B * N, D, pad_left + q_len)
            else:
                buf = h_res.transpose(2, 3).reshape(B * N, D, q_len)
            conv_out = layer.conv.dwconv.conv(buf).transpose(1, 2).reshape(B, N, q_len, D)
            h = h_res + layer.conv.pw2(F.gelu(layer.conv.pw1(layer.conv.norm(conv_out))))

        return h[:, :, :C, :].reshape(B, N * C, D)[:, :T, :]


class AcousticOfflineWrapper(nn.Module):
    """离线声学模型 ONNX 包装器：直接输出反归一化后的 raw log-mel [1, n_mels, F] 与时长 [1, T]。"""

    def __init__(self, model: lib.AetherAcousticModel, mel_mean: np.ndarray, mel_std: np.ndarray):
        super().__init__()
        self.model = model
        self.use_chunk_onnx = isinstance(model.backbone, lib.ChunkLookaheadBackbone)
        if self.use_chunk_onnx:
            self.chunk_bb = ChunkBackboneOfflineONNX(model.backbone)
        self.expander = FrameExpanderONNX(model.frame_phase_proj)
        self.register_buffer("mel_mean", torch.tensor(mel_mean, dtype=torch.float32).view(1, -1, 1), persistent=False)
        self.register_buffer("mel_std", torch.tensor(mel_std, dtype=torch.float32).view(1, -1, 1), persistent=False)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        boundary_ids: torch.Tensor,
        speed: torch.Tensor,
    ):
        x_emb = self.model.embed_text(phoneme_ids, lang_ids, text_mask=None, boundary_ids=boundary_ids)
        h_txt = self.chunk_bb(x_emb) if self.use_chunk_onnx else self.model.backbone(x_emb)
        log_dur = self.model.dur_predictor(h_txt)
        dur = torch.clamp(torch.round((torch.exp(log_dur) - 1.0) / speed), min=1.0, max=60.0).to(torch.long)
        h_frame = self.expander(h_txt, dur)
        mel_norm, _, _, _ = self.model.decode_from_frames(h_frame)
        mel = mel_norm * self.mel_std + self.mel_mean
        return mel, dur


def _harmonic_oscillate_onnx(
    harm: lib.CausalHarmonicSource,
    fft: CooleyTukeyFFT1D,
    log_f0_pred: torch.Tensor,
    uv_logit: torch.Tensor,
    phi_init: torch.Tensor = None,
    prev_f0: torch.Tensor = None,
):
    """连续相位 24 阶切比雪夫 STFT 谐波激励源的纯实数 ONNX 等价实现（float64 梯形积分）。"""
    f0_in = log_f0_pred.float()
    uv_prob = torch.sigmoid(uv_logit.float())
    f0_hz = torch.exp(f0_in.clamp(min=math.log(55.0), max=math.log(620.0))).double()
    if prev_f0 is None:
        first_prev = f0_hz[:, :1]
    else:
        first_prev = torch.where(prev_f0 > 0.0, prev_f0, f0_hz[:, :1])
    f0_prev_seq = torch.cat([first_prev, f0_hz[:, :-1]], dim=1)

    delta_phi = 0.5 * (f0_prev_seq + f0_hz) * (float(harm.hop_length) / float(harm.sample_rate))
    csum = torch.cumsum(delta_phi, dim=1)
    if phi_init is not None:
        csum = phi_init + csum
    # 非负标量下 x - floor(x) 严格等于 fmod(x, 1.0)，且全 ONNX 后端原生支持
    phi = csum - torch.floor(csum)
    new_phi = phi[:, -1:]
    new_prev = f0_hz[:, -1:]

    phase_raw = phi.unsqueeze(-1) + f0_hz.unsqueeze(-1) * harm.time_offsets
    phase_grid = phase_raw - torch.floor(phase_raw)
    theta = (2.0 * math.pi * phase_grid).to(torch.float32)
    c1 = torch.cos(theta)
    c_prev2 = torch.ones_like(c1)
    c_prev1 = c1
    s_time = torch.zeros_like(c1)
    f0_f32 = f0_hz.to(torch.float32).unsqueeze(-1)
    nyq_lim = 0.45 * float(harm.sample_rate)
    for h_ord in range(1, harm.n_harmonics + 1):
        c_cur = c1 if h_ord == 1 else (2.0 * c1 * c_prev1 - c_prev2)
        if h_ord > 1:
            c_prev2, c_prev1 = c_prev1, c_cur
        nyq_m = (f0_f32 * float(h_ord) < nyq_lim).to(torch.float32)
        s_time = s_time + (harm.harm_amps[h_ord - 1] * nyq_m) * c_cur

    harm_re, harm_im = fft.rfft(s_time * harm.hann_win)
    uv_scale = uv_prob.unsqueeze(-1)
    harm_re = harm_re * uv_scale
    harm_im = harm_im * uv_scale

    angles = (2.0 * math.pi * phi.unsqueeze(-1) * harm.cond_orders).to(torch.float32)
    harms = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) * uv_scale
    f0_norm = ((f0_in - 5.3) / 0.5).unsqueeze(-1)
    feats = torch.cat([harms, f0_norm, uv_scale], dim=-1)
    harm_in = harm.in_proj(feats)
    return harm_in, harm_re, harm_im, new_phi, new_prev


def _istft_head_frames_onnx(
    head: lib.StreamingISTFTHead,
    fft: CooleyTukeyFFT1D,
    x: torch.Tensor,
    harm_re: torch.Tensor,
    harm_im: torch.Tensor,
) -> torch.Tensor:
    """谐波源-滤波器 + 噪声双分支复频域合成 -> 纯实数 irfft + Hann 窗，输出 [B, F, n_fft]。"""
    mag_h, p_h, mag_n, p_n = head.out(x.float()).chunk(4, dim=-1)
    amp_n = torch.exp(mag_n.clamp(min=-12.0, max=4.6))
    spec_re = amp_n * torch.cos(p_n)
    spec_im = amp_n * torch.sin(p_n)

    gain_h = torch.exp(mag_h.clamp(min=-8.0, max=4.0))
    filt_re = gain_h * torch.cos(p_h)
    filt_im = gain_h * torch.sin(p_h)

    spec_re = spec_re + (harm_re * filt_re - harm_im * filt_im)
    spec_im = spec_im + (harm_re * filt_im + harm_im * filt_re)
    return fft.irfft(spec_re, spec_im) * head.window[None, None, :]


class VocoderOfflineWrapper(nn.Module):
    """离线 CausalVocosLite ONNX 包装器：mel [1, n_mels, F] -> wav [1, F * hop_length]。"""

    def __init__(self, voc: lib.CausalVocosLite):
        super().__init__()
        if voc.n_fft % voc.hop_length != 0:
            raise ValueError(f"要求 n_fft ({voc.n_fft}) 可被 hop_length ({voc.hop_length}) 整除")
        self.voc = voc
        self.fft = CooleyTukeyFFT1D(voc.n_fft)
        self.n_fft = voc.n_fft
        self.hop_length = voc.hop_length
        self.overlap_blocks = voc.n_fft // voc.hop_length

    def _ola_offline(self, frames: torch.Tensor) -> torch.Tensor:
        # 严格等价于 StreamingISTFTHead.forward 中的 F.fold + Hann^2 归一化，零 col2im 算子
        B, F_len, _ = frames.shape
        K, hop = self.overlap_blocks, self.hop_length
        chunks = frames.reshape(B, F_len, K, hop)
        out_blocks = None
        for k in range(K):
            parts = []
            if k > 0:
                parts.append(torch.zeros((B, k, hop), device=frames.device, dtype=frames.dtype))
            parts.append(chunks[:, :, k, :])
            if K - 1 - k > 0:
                parts.append(torch.zeros((B, K - 1 - k, hop), device=frames.device, dtype=frames.dtype))
            shifted = torch.cat(parts, dim=1)
            out_blocks = shifted if out_blocks is None else (out_blocks + shifted)
        out = out_blocks.reshape(B, (F_len + K - 1) * hop)

        w2 = (self.voc.head.window ** 2).reshape(1, 1, K, hop).expand(1, F_len, K, hop)
        env_blocks = None
        for k in range(K):
            parts = []
            if k > 0:
                parts.append(torch.zeros((1, k, hop), device=frames.device, dtype=frames.dtype))
            parts.append(w2[:, :, k, :])
            if K - 1 - k > 0:
                parts.append(torch.zeros((1, K - 1 - k, hop), device=frames.device, dtype=frames.dtype))
            shifted = torch.cat(parts, dim=1)
            env_blocks = shifted if env_blocks is None else (env_blocks + shifted)
        env = env_blocks.reshape(1, (F_len + K - 1) * hop)

        left_trim = self.n_fft // 2
        end_trim = left_trim + F_len * hop
        return out[:, left_trim:end_trim] / env[:, left_trim:end_trim].clamp(min=1e-6)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.voc._norm_input(mel)
        h = self.voc.norm_in(self.voc.embed(x).transpose(1, 2))
        hs = self.voc.harm_source
        p_hid = F.silu(hs.pitch_norm1(hs.pitch_conv1(h.transpose(1, 2)).transpose(1, 2)))
        p_hid = p_hid + F.silu(hs.pitch_norm2(hs.pitch_conv2(p_hid.transpose(1, 2)).transpose(1, 2)))
        p_out = hs.pitch_out(p_hid.float())
        harm_in, harm_re, harm_im, _, _ = _harmonic_oscillate_onnx(hs, self.fft, p_out[..., 0], p_out[..., 1])
        h = h + harm_in
        for blk in self.voc.blocks:
            h = blk(h)
        frames = _istft_head_frames_onnx(self.voc.head, self.fft, self.voc.norm_out(h), harm_re, harm_im)
        return self._ola_offline(frames)


# ==============================================================================
# 4. 三段式流式 ONNX 包装器：text_step.onnx / frame_step.onnx / vocoder_step.onnx
# ==============================================================================
class TextStepWrapper(nn.Module):
    """流式阶段 1 (text_step.onnx)：分块深前瞻文本编码 + 时长预测 + 帧相位扩展。

    使用固定尺寸右对齐 KV Cache ([1, H, W, dh]) + 标量 past_len，规避空张量与动态位置溢出，
    与 ChunkLookaheadBackbone.forward_step / infer_stream_step 严格数学等价。
    """

    def __init__(self, model: lib.AetherAcousticModel):
        super().__init__()
        assert isinstance(model.backbone, lib.ChunkLookaheadBackbone), "text_step.onnx 仅支持 ChunkLookaheadBackbone"
        self.model = model
        self.backbone = model.backbone
        self.n_layers = len(self.backbone.layers)
        self.window_size = self.backbone.layers[0].window_size
        self.max_win = self.backbone.chunk_size + self.backbone.lookahead
        self.n_dur_layers = len(model.dur_predictor.dw_convs)
        self.expander = FrameExpanderONNX(model.frame_phase_proj)

        cos, sin = self.backbone.layers[0]._get_local_rope(self.window_size + self.max_win, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos.clone(), persistent=False)
        self.register_buffer("rope_sin", sin.clone(), persistent=False)
        self.register_buffer("w_range", torch.arange(self.window_size, dtype=torch.long), persistent=False)
        pad_left = self.backbone.layers[0].conv.dwconv.pad_left
        self.register_buffer("conv_range", torch.arange(pad_left, dtype=torch.long), persistent=False)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        boundary_ids: torch.Tensor,
        commit_len: torch.Tensor,
        speed: torch.Tensor,
        past_len: torch.Tensor,
        *caches,
    ):
        # caches 顺序: k_0..k_{L-1}, v_0..v_{L-1}, conv_0..conv_{L-1}, dur_0..dur_{D-1}
        nL = self.n_layers
        k_caches = caches[0:nL]
        v_caches = caches[nL : 2 * nL]
        c_caches = caches[2 * nL : 3 * nL]
        d_caches = caches[3 * nL :]

        h = self.model.embed_text(phoneme_ids, lang_ids, text_mask=None, boundary_ids=boundary_ids)
        B, M, D = h.shape
        W = self.window_size
        c_scalar = commit_len[0]
        p_scalar = past_len[0]

        past_valid = (self.w_range >= (W - p_scalar)).unsqueeze(0)                    # [1, W]
        key_valid = torch.cat([past_valid, torch.ones((B, M), dtype=torch.bool, device=h.device)], dim=1)  # [1, W + M]
        idx_k = c_scalar + self.w_range                                               # [W]
        idx_c = c_scalar + self.conv_range                                            # [pad_left]
        cos_q = self.rope_cos[None, None, W : W + M, :]
        sin_q = self.rope_sin[None, None, W : W + M, :]
        cos_k = self.rope_cos[None, None, : W + M, :]
        sin_k = self.rope_sin[None, None, : W + M, :]

        new_ks, new_vs, new_cs = [], [], []
        for layer, k_st, v_st, c_st in zip(self.backbone.layers, k_caches, v_caches, c_caches):
            H, dh = layer.n_heads, layer.head_dim
            qkv = layer.qkv(layer.norm1(h)).reshape(B, M, 3, H, dh)
            q = qkv[:, :, 0].transpose(1, 2)
            k_raw = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)

            k_cat_raw = torch.cat([k_st, k_raw], dim=2)                               # [1, H, W + M, dh]
            v_all = torch.cat([v_st, v], dim=2)

            q_rot = _apply_rope_onnx(q, cos_q, sin_q)
            k_all = _apply_rope_onnx(k_cat_raw, cos_k, sin_k)

            scores = torch.matmul(q_rot, k_all.transpose(-2, -1)) * (1.0 / math.sqrt(float(dh)))
            scores = torch.where(key_valid.unsqueeze(1).unsqueeze(2), scores, torch.full_like(scores, -1e9))
            out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).reshape(B, M, D)
            h_res = h + layer.out_proj(out)

            buf = torch.cat([c_st, h_res.transpose(1, 2)], dim=-1)                    # [1, D, pad_left + M]
            conv_out = layer.conv.dwconv.conv(buf).transpose(1, 2)
            h = h_res + layer.conv.pw2(F.gelu(layer.conv.pw1(layer.conv.norm(conv_out))))

            new_ks.append(k_cat_raw.index_select(2, idx_k))
            new_vs.append(v_all.index_select(2, idx_k))
            new_cs.append(buf.index_select(2, idx_c))

        h_commit = h[:, :c_scalar, :]
        new_past_len = torch.clamp(past_len + commit_len, max=W)

        # 时长因果卷积预测 + 帧扩展
        hd = h_commit
        new_ds = []
        for dw, ln, pw, dst in zip(
            self.model.dur_predictor.dw_convs,
            self.model.dur_predictor.norms,
            self.model.dur_predictor.pws,
            d_caches,
        ):
            buf_d = torch.cat([dst, hd.transpose(1, 2)], dim=-1)
            y_d = dw.conv(buf_d).transpose(1, 2)
            hd = hd + F.silu(pw(ln(y_d)))
            new_ds.append(buf_d[:, :, -dw.pad_left :])
        log_dur = self.model.dur_predictor.proj(hd).squeeze(-1)
        durations = torch.clamp(torch.round((torch.exp(log_dur) - 1.0) / speed), min=1.0, max=60.0).to(torch.long)
        h_frame = self.expander(h_commit, durations)
        return (h_frame, durations, new_past_len, *new_ks, *new_vs, *new_cs, *new_ds)


class FrameStepWrapper(nn.Module):
    """流式阶段 2 (frame_step.onnx)：frame_pre 滑窗 + F0/UV/Energy + 6 层帧解码 + Mel 反归一化。

    调用方维护长度至少为 k_left + 1 + frame_lookahead 的 FIFO 帧窗口 h_frame_win，
    每次输出 n_emit = len(h_frame_win) - k_left - frame_lookahead 帧 raw log-mel。
    """

    def __init__(self, model: lib.AetherAcousticModel, mel_mean: np.ndarray, mel_std: np.ndarray):
        super().__init__()
        self.model = model
        self.k_left = model.frame_pre.k_left
        self.lookahead = model.frame_pre.lookahead
        self.n_f0 = len(model.f0_uv_predictor.dw_convs)
        self.n_en = len(model.energy_predictor.dw_convs)
        self.n_dec = len(model.frame_decoder)
        self.register_buffer("mel_mean", torch.tensor(mel_mean, dtype=torch.float32).view(1, -1, 1), persistent=False)
        self.register_buffer("mel_std", torch.tensor(mel_std, dtype=torch.float32).view(1, -1, 1), persistent=False)

    def forward(self, h_frame_win: torch.Tensor, *caches):
        f0_caches = caches[0 : self.n_f0]
        en_caches = caches[self.n_f0 : self.n_f0 + self.n_en]
        dec_caches = caches[self.n_f0 + self.n_en :]

        conv_out = self.model.frame_pre.dwconv(h_frame_win.transpose(1, 2))
        if self.lookahead > 0:
            residual = h_frame_win[:, self.k_left : h_frame_win.shape[1] - self.lookahead, :]
        else:
            residual = h_frame_win[:, self.k_left :, :]
        h_pre = self.model.frame_pre._apply_body(conv_out, residual)

        hf = h_pre
        new_f0 = []
        for dw, ln, pw, st in zip(
            self.model.f0_uv_predictor.dw_convs,
            self.model.f0_uv_predictor.norms,
            self.model.f0_uv_predictor.pws,
            f0_caches,
        ):
            buf = torch.cat([st, hf.transpose(1, 2)], dim=-1)
            y = dw.conv(buf).transpose(1, 2)
            hf = hf + F.silu(pw(ln(y)))
            new_f0.append(buf[:, :, -dw.pad_left :])
        f0_uv = self.model.f0_uv_predictor.proj(hf)
        f0_pred, uv_prob = f0_uv[..., 0], torch.sigmoid(f0_uv[..., 1])

        he = h_pre
        new_en = []
        for dw, ln, pw, st in zip(
            self.model.energy_predictor.dw_convs,
            self.model.energy_predictor.norms,
            self.model.energy_predictor.pws,
            en_caches,
        ):
            buf = torch.cat([st, he.transpose(1, 2)], dim=-1)
            y = dw.conv(buf).transpose(1, 2)
            he = he + F.silu(pw(ln(y)))
            new_en.append(buf[:, :, -dw.pad_left :])
        e_pred = self.model.energy_predictor.proj(he).squeeze(-1)

        h_dec = (
            h_pre
            + self.model.f0_emb((f0_pred * uv_prob).unsqueeze(-1))
            + self.model.uv_emb(uv_prob.unsqueeze(-1))
            + self.model.energy_emb(e_pred.unsqueeze(-1))
        )
        new_dec = []
        for blk, st in zip(self.model.frame_decoder, dec_caches):
            buf = torch.cat([st, h_dec.transpose(1, 2)], dim=-1)
            y = blk.dwconv.conv(buf).transpose(1, 2)
            h_dec = h_dec + blk.pw2(F.gelu(blk.pw1(blk.norm(y))))
            new_dec.append(buf[:, :, -blk.dwconv.pad_left :])

        mel_norm = self.model.mel_out(h_dec).transpose(1, 2)
        mel = mel_norm * self.mel_std + self.mel_mean
        return (mel, *new_f0, *new_en, *new_dec)


class VocoderStepWrapper(nn.Module):
    """流式阶段 3 (vocoder_step.onnx)：Lookahead Mel 窗 -> 谐波源 + 8 层 ConvNeXt -> 窗函数时域帧。

    调用方维护长度至少为 k_left + 1 + lookahead 的 raw log-mel 滑窗（首尾以 mel_mean 填充，
    归一化后精确为 0），输出 [1, n_emit, n_fft] 的加窗时域帧供轻量 OLA 实时拼接 PCM。
    """

    def __init__(self, voc: lib.CausalVocosLite):
        super().__init__()
        self.voc = voc
        self.fft = CooleyTukeyFFT1D(voc.n_fft)

    def forward(
        self,
        mel_win: torch.Tensor,
        harm_conv1_cache: torch.Tensor,
        harm_conv2_cache: torch.Tensor,
        phi: torch.Tensor,
        prev_f0: torch.Tensor,
        *block_caches,
    ):
        x = self.voc._norm_input(mel_win)
        h = self.voc.norm_in(self.voc.embed.conv(x).transpose(1, 2))

        hs = self.voc.harm_source
        b1 = torch.cat([harm_conv1_cache, h.transpose(1, 2)], dim=-1)
        p1 = hs.pitch_conv1.conv(b1).transpose(1, 2)
        new_hc1 = b1[:, :, -hs.pitch_conv1.pad_left :]
        p_hid = F.silu(hs.pitch_norm1(p1))

        b2 = torch.cat([harm_conv2_cache, p_hid.transpose(1, 2)], dim=-1)
        p2 = hs.pitch_conv2.conv(b2).transpose(1, 2)
        new_hc2 = b2[:, :, -hs.pitch_conv2.pad_left :]
        p_hid = p_hid + F.silu(hs.pitch_norm2(p2))

        p_out = hs.pitch_out(p_hid.float())
        harm_in, harm_re, harm_im, new_phi, new_prev_f0 = _harmonic_oscillate_onnx(
            hs, self.fft, p_out[..., 0], p_out[..., 1], phi_init=phi, prev_f0=prev_f0
        )
        h = h + harm_in

        new_blks = []
        for blk, st in zip(self.voc.blocks, block_caches):
            buf = torch.cat([st, h.transpose(1, 2)], dim=-1)
            y = blk.dwconv.conv(buf).transpose(1, 2)
            h = h + blk.pw2(F.gelu(blk.pw1(blk.norm(y))))
            new_blks.append(buf[:, :, -blk.dwconv.pad_left :])

        time_frames = _istft_head_frames_onnx(self.voc.head, self.fft, self.voc.norm_out(h), harm_re, harm_im)
        return (time_frames, new_hc1, new_hc2, new_phi, new_prev_f0, *new_blks)


# ==============================================================================
# 5. 跨版本 ONNX 导出助手 & 可选动态 INT8 量化
# ==============================================================================
def _export_onnx(
    module: nn.Module,
    args: tuple,
    out_path: Path,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int = None,
    exporter: str = "auto",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(torch.onnx.export)
    has_dynamo = "dynamo" in sig.parameters
    has_ext = "external_data" in sig.parameters

    # 默认优先 legacy (opset 17)：速度极快、零外部数据分片、支持数据相关动态长度且无需 onnxscript
    use_dynamo = (exporter == "dynamo") and has_dynamo
    eff_opset = opset if opset is not None else (18 if use_dynamo else 17)

    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": eff_opset,
        "do_constant_folding": True,
    }
    if has_dynamo:
        kwargs["dynamo"] = use_dynamo
    if use_dynamo:
        if has_ext:
            kwargs["external_data"] = False
        kwargs["dynamic_axes"] = dynamic_axes
    else:
        kwargs["dynamic_axes"] = dynamic_axes

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*TracerWarning.*")
        warnings.filterwarnings("ignore", message=".*Constant folding - Only steps=1.*")
        torch.onnx.export(module.eval(), args, str(out_path), **kwargs)

    # 仅使用 onnx.checker 做结构完整性校验
    try:
        import onnx

        onnx.checker.check_model(str(out_path))
    except ImportError:
        pass
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  [Export] {out_path.name:20s} | opset={eff_opset} | {size_mb:6.2f} MB -> {out_path}")


def quantize_onnx_int8(src_path: Path) -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst_path = src_path.with_name(src_path.stem + ".int8.onnx")

    # 严格遵守 AetherTTS 精度策略：
    # - 仅量化主干 Linear (MatMul) 权重；
    # - 保护非权重 MatMul（Cooley-Tukey FFT 与 Attention Q@K/A@V）及敏感输出头
    #   （声码器 pitch_out / iSTFT out，声学 mel_out / dur/f0/energy proj）维持 FP32/FP64。
    model_proto = onnx.load(str(src_path))
    inits = {t.name for t in model_proto.graph.initializer}
    sensitive_tokens = ("pitch_out", "/out/", "mel_out", "/proj/", "/proj_1/")
    exclude = []
    for node in model_proto.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        has_init = any(inp in inits for inp in node.input)
        is_sensitive = any(tok in node.name for tok in sensitive_tokens)
        if (not has_init) or is_sensitive:
            exclude.append(node.name)

    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=exclude,
        weight_type=QuantType.QInt8,
    )
    size_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  [INT8]   {dst_path.name:22s} | {size_mb:6.2f} MB (FP32 保护节点={len(exclude):2d}) -> {dst_path}")
    return dst_path


# ==============================================================================
# 6. 数值一致性门禁校验（PyTorch <-> ONNX Runtime，含离线 <-> 流式交叉比对）
# ==============================================================================
def _ort_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


@torch.no_grad()
def verify_exports(
    out_dir: Path,
    meta: dict,
    acoustic_model: lib.AetherAcousticModel = None,
    vocoder_model: lib.CausalVocosLite = None,
):
    print("\n" + "=" * 76)
    print("[Verify] PyTorch <-> ONNX Runtime 数值一致性校验（多动态长度 + 离线/流式交叉验证）")
    print("=" * 76)

    # 0) 校验纯 Python 前端与 aethertts_lib 完全一致
    pure_fe = PureTextFrontend(vocab=meta["vocab"])
    for sample in ("你好，请问今天北京的天气怎么样？", "帮我把 Wi-Fi 切换到 5G，一会儿见！"):
        assert pure_fe.text_to_ids(sample) == lib.FRONTEND.text_to_ids(sample), f"前端输出不一致: {sample}"
    print("  [PASS] text_frontend.py <-> aethertts_lib.py 音素/语种/韵律边界 100% 一致")

    mel_mean = np.asarray(meta["mel_mean"], dtype=np.float32).reshape(1, -1, 1)
    mel_std = np.asarray(meta["mel_std"], dtype=np.float32).reshape(1, -1, 1)

    sample_texts = [
        "你好！",
        "你好，请问今天北京的天气怎么样？",
        "展览馆门口有一点儿花儿，一会儿我们把系统切换到最新模式。",
    ]

    # 1) 离线声学校验
    mel_ref_for_voc = None
    if acoustic_model is not None and (out_dir / "acoustic.onnx").exists():
        ac_sess = _ort_session(out_dir / "acoustic.onnx")
        max_mel_err = 0.0
        for txt in sample_texts:
            pids, lids, bids = pure_fe.text_to_ids(txt)
            for spd in (1.0, 1.2):
                pt = torch.tensor([pids], dtype=torch.long)
                lt = torch.tensor([lids], dtype=torch.long)
                bt = torch.tensor([bids], dtype=torch.long)
                mel_pt_norm, dur_pt, _, _ = acoustic_model.infer_offline(pt, lt, speed=spd, boundary_ids=bt)
                mel_pt = mel_pt_norm.numpy() * mel_std + mel_mean
                mel_ort, dur_ort = ac_sess.run(
                    None,
                    {
                        "phoneme_ids": pt.numpy(),
                        "lang_ids": lt.numpy(),
                        "boundary_ids": bt.numpy(),
                        "speed": np.array([spd], dtype=np.float32),
                    },
                )
                assert np.array_equal(dur_pt.numpy(), dur_ort), f"Duration mismatch for T={len(pids)}, speed={spd}"
                err = float(np.max(np.abs(mel_pt - mel_ort)))
                max_mel_err = max(max_mel_err, err)
                if spd == 1.0 and mel_ref_for_voc is None:
                    mel_ref_for_voc = mel_ort
        print(f"  [PASS] acoustic.onnx (离线, 多长度 T & 变语速) | max_abs_err = {max_mel_err:.2e}")
        assert max_mel_err < 1e-3, f"acoustic.onnx 数值偏差过大: {max_mel_err}"

    # 2) 离线声码器校验
    if vocoder_model is not None and (out_dir / "vocoder.onnx").exists():
        voc_sess = _ort_session(out_dir / "vocoder.onnx")
        max_wav_err, max_rms_err = 0.0, 0.0
        rng = np.random.default_rng(42)
        test_mels = [
            (rng.standard_normal((1, meta["n_mels"], n_f)).astype(np.float32) * mel_std + mel_mean)
            for n_f in (12, 37, 85)
        ]
        if mel_ref_for_voc is not None:
            test_mels.append(mel_ref_for_voc)
        for m_np in test_mels:
            wav_pt = vocoder_model(torch.from_numpy(m_np)).numpy()
            (wav_ort,) = voc_sess.run(None, {"mel": m_np})
            diff = np.abs(wav_pt - wav_ort)
            max_wav_err = max(max_wav_err, float(diff.max()))
            max_rms_err = max(max_rms_err, float(np.sqrt(np.mean(diff ** 2))))
        print(f"  [PASS] vocoder.onnx  (离线, F=12/37/85+实谱) | max_abs = {max_wav_err:.2e}, rms = {max_rms_err:.2e}")
        assert max_rms_err < 1e-3, f"vocoder.onnx RMS 偏差过大: {max_rms_err}"

    # 3) 三段流式校验（使用 infer_onnx.py 中的流式执行引擎，顺便联调推理代码！）
    if (out_dir / "text_step.onnx").exists() and (out_dir / "frame_step.onnx").exists():
        from scripts.infer_onnx import AetherONNXStreamingEngine

        has_voc_stream = (out_dir / "vocoder_step.onnx").exists() and vocoder_model is not None
        engine = AetherONNXStreamingEngine(out_dir, use_vocoder=has_voc_stream, num_threads=2)

        txt = sample_texts[2]
        pids, lids, bids = pure_fe.text_to_ids(txt)
        pt = torch.tensor([pids], dtype=torch.long)
        lt = torch.tensor([lids], dtype=torch.long)
        bt = torch.tensor([bids], dtype=torch.long)

        # PyTorch 离线基准
        mel_pt_norm, _, _, _ = acoustic_model.infer_offline(pt, lt, speed=1.0, boundary_ids=bt)
        mel_pt = mel_pt_norm.numpy() * mel_std + mel_mean

        wav_stream, mel_stream, stats_s = engine.synthesize_ids(pids, lids, bids, speed=1.0)
        mel_stream_err = float(np.max(np.abs(mel_pt - mel_stream)))
        print(
            f"  [PASS] text_step + frame_step (流式 vs PyTorch 离线, T={len(pids)}, F={mel_stream.shape[-1]}) "
            f"| max_mel_err = {mel_stream_err:.2e}"
        )
        assert mel_stream_err < 1e-3, f"流式声学 ONNX 偏差过大: {mel_stream_err}"

        if has_voc_stream:
            wav_pt = vocoder_model(torch.from_numpy(mel_pt)).numpy().squeeze(0)
            min_l = min(len(wav_pt), len(wav_stream))
            diff = np.abs(wav_pt[:min_l] - wav_stream[:min_l])
            rms_err = float(np.sqrt(np.mean(diff ** 2)))
            print(
                f"  [PASS] vocoder_step  (端到端三段流式 vs PyTorch 离线) "
                f"| len={len(wav_stream)} (ref={len(wav_pt)}) | max_abs={diff.max():.2e}, rms={rms_err:.2e} "
                f"| TTFA={stats_s['ttfa_ms']:.1f}ms, RTF={stats_s['rtf']:.3f}"
            )
            assert len(wav_stream) == len(wav_pt), f"流式波形长度不一致: {len(wav_stream)} vs {len(wav_pt)}"
            assert rms_err < 2e-3, f"三段流式端到端波形 RMS 偏差过大: {rms_err}"

    print("=" * 76)
    print("[Verify] 全部 ONNX 数值一致性校验通过！")


# ==============================================================================
# 7. 主导出流程
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="导出 AetherTTS (声学模型 + 谐波源因果声码器) 到 ONNX")
    ap.add_argument("--root", default="./aethertts_exp", help="训练产物根目录（用于自动发现 checkpoint）")
    ap.add_argument("--profile", default="standard_phase1", help="训练档位名称（如 full_quality / standard_phase1）")
    ap.add_argument("--dataset", default="bznsyp", help="数据集标识")
    ap.add_argument("--model", default="E3_chunk_attn_C4_L8", help="声学实验目录前缀")
    ap.add_argument("--acoustic-ckpt", default="", help="手动指定声学模型 checkpoint 路径 (best.pt / latest.pt)")
    ap.add_argument("--vocoder-ckpt", default="", help="手动指定声码器 checkpoint 路径 (final.pt / latest.pt)")
    ap.add_argument("--no-acoustic", action="store_true", help="跳过声学模型导出")
    ap.add_argument("--no-vocoder", action="store_true", help="跳过声码器导出")
    ap.add_argument("--dummy", action="store_true", help="使用随机初始化的 a9 模型导出（用于无权重环境冒烟测试）")
    ap.add_argument("--mode", default="both", choices=["offline", "streaming", "both"], help="导出模式（默认 both）")
    ap.add_argument("--out-dir", default="./onnx_export", help="ONNX 输出目录")
    ap.add_argument("--opset", type=int, default=None, help="指定 ONNX opset（默认 legacy=17, dynamo=18）")
    ap.add_argument("--exporter", default="auto", choices=["auto", "legacy", "dynamo"], help="PyTorch ONNX 导出后端")
    ap.add_argument("--quantize-int8", action="store_true", help="额外生成权重 INT8 动态量化版本 (*.int8.onnx)")
    ap.add_argument("--skip-verify", action="store_true", help="跳过导出后的 ONNX Runtime 数值等价校验")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    acoustic_model = None
    vocoder_model = None
    cfg = dict(lib.CFG)
    stats = None
    vocab = list(lib.FRONTEND.vocab)

    if args.dummy:
        print("[Load] --dummy 模式：初始化随机权重 AetherTTS a9 声学模型与 CausalVocosLite")
        if not args.no_acoustic:
            acoustic_model = lib.AetherAcousticModel(
                vocab_size=len(vocab),
                backbone_type="chunk_attn",
                d_model=cfg["d_model"],
                n_mels=cfg["n_mels"],
                k_left=cfg["k_left"],
                lookahead=8,
                swa_window=cfg["swa_window"],
                chunk_size=cfg["chunk_attn_size"],
                frame_lookahead=cfg["frame_lookahead"],
                dropout=0.0,
            ).export_inference_copy().float().eval()
        if not args.no_vocoder:
            vocoder_model = lib.CausalVocosLite(
                n_mels=cfg["n_mels"],
                dim=cfg["vocoder_dim"],
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                lookahead=cfg["vocoder_lookahead"],
                sample_rate=cfg["sample_rate"],
            ).float().eval()
            vocoder_model.set_mel_stats(np.full(cfg["n_mels"], -4.5, dtype=np.float32), np.full(cfg["n_mels"], 2.2, dtype=np.float32))
        stats = {
            "mel_mean": [-4.5] * cfg["n_mels"],
            "mel_std": [2.2] * cfg["n_mels"],
        }
    else:
        if not args.no_acoustic:
            ac_path = find_acoustic_ckpt(root, args.model, args.profile, args.dataset, args.acoustic_ckpt)
            acoustic_model, ac_cfg, stats, vocab, ckpt = load_acoustic_model(ac_path)
            cfg.update(ac_cfg)
            print(
                f"[Load] 声学权重: {ac_path} | backbone={acoustic_model.backbone_type} "
                f"| params={lib.count_params_m(acoustic_model):.2f}M | step={ckpt.get('step')}"
            )
        if not args.no_vocoder:
            voc_path = find_vocoder_ckpt(root, args.profile, args.dataset, args.vocoder_ckpt)
            if voc_path is None:
                print("[Warn] 未发现因果声码器权重 (final.pt / latest.pt)，本次仅导出声学模型。如需导出声码器请传 --vocoder-ckpt。")
            else:
                vocoder_model = load_vocoder_model(voc_path, cfg)
                print(
                    f"[Load] 声码器权重: {voc_path} | dim={vocoder_model.dim}, lookahead={vocoder_model.lookahead} "
                    f"| params={lib.count_params_m(vocoder_model):.2f}M"
                )

    if acoustic_model is None and vocoder_model is None:
        raise RuntimeError("声学模型与声码器均未加载，无可导出模型。")

    # 统一 Mel 反归一化统计量（优先取声学 checkpoint 的 stats，其次回退到声码器内置 buffer）
    n_mels = acoustic_model.n_mels if acoustic_model is not None else vocoder_model.mel_mean.shape[1]
    if stats is not None and "mel_mean" in stats and "mel_std" in stats:
        mel_mean = np.asarray(stats["mel_mean"], dtype=np.float32).reshape(-1)
        mel_std = np.asarray(stats["mel_std"], dtype=np.float32).reshape(-1)
    elif vocoder_model is not None:
        mel_mean = vocoder_model.mel_mean.detach().cpu().numpy().reshape(-1)
        mel_std = vocoder_model.mel_std.detach().cpu().numpy().reshape(-1)
    else:
        mel_mean = np.zeros(n_mels, dtype=np.float32)
        mel_std = np.ones(n_mels, dtype=np.float32)

    want_offline = args.mode in ("offline", "both")
    want_streaming = args.mode in ("streaming", "both")
    if want_streaming and acoustic_model is not None and acoustic_model.backbone_type != "chunk_attn":
        if args.mode == "streaming":
            raise ValueError(f"流式三段导出仅支持 chunk_attn 主干，当前为 {acoustic_model.backbone_type}")
        print(f"[Warn] 当前声学主干为 {acoustic_model.backbone_type}（非 chunk_attn），自动跳过流式导出。")
        want_streaming = False

    exported_files = []
    meta = {
        "format_version": "aethertts_onnx_v1",
        "sample_rate": int(cfg.get("sample_rate", 24000)),
        "n_fft": int(vocoder_model.n_fft if vocoder_model is not None else cfg.get("n_fft", 1024)),
        "hop_length": int(vocoder_model.hop_length if vocoder_model is not None else cfg.get("hop_length", 256)),
        "n_mels": int(n_mels),
        "mel_mean": mel_mean.tolist(),
        "mel_std": mel_std.tolist(),
        "vocab": vocab,
        "modes": [],
    }

    # --- 1) 导出离线两段式 ---
    if want_offline:
        print("\n[Stage] 导出离线 ONNX 模型 (acoustic.onnx / vocoder.onnx)...")
        if acoustic_model is not None:
            ac_wrap = AcousticOfflineWrapper(acoustic_model, mel_mean, mel_std).eval()
            T_demo = 16
            dummy_pids = torch.randint(1, len(vocab), (1, T_demo), dtype=torch.long)
            dummy_lids = torch.ones((1, T_demo), dtype=torch.long)
            dummy_bids = torch.zeros((1, T_demo), dtype=torch.long)
            dummy_spd = torch.tensor([1.0], dtype=torch.float32)
            ac_onnx = out_dir / "acoustic.onnx"
            _export_onnx(
                ac_wrap,
                (dummy_pids, dummy_lids, dummy_bids, dummy_spd),
                ac_onnx,
                input_names=["phoneme_ids", "lang_ids", "boundary_ids", "speed"],
                output_names=["mel", "durations"],
                dynamic_axes={
                    "phoneme_ids": {1: "T"},
                    "lang_ids": {1: "T"},
                    "boundary_ids": {1: "T"},
                    "mel": {2: "F"},
                    "durations": {1: "T"},
                },
                opset=args.opset,
                exporter=args.exporter,
            )
            exported_files.append(ac_onnx)
        if vocoder_model is not None:
            voc_wrap = VocoderOfflineWrapper(vocoder_model).eval()
            dummy_mel = torch.randn(1, n_mels, 24, dtype=torch.float32)
            voc_onnx = out_dir / "vocoder.onnx"
            _export_onnx(
                voc_wrap,
                (dummy_mel,),
                voc_onnx,
                input_names=["mel"],
                output_names=["wav"],
                dynamic_axes={"mel": {2: "F"}, "wav": {1: "S"}},
                opset=args.opset,
                exporter=args.exporter,
            )
            exported_files.append(voc_onnx)
        meta["modes"].append("offline")

    # --- 2) 导出三段式流式 ---
    if want_streaming:
        print("\n[Stage] 导出三段式流式 ONNX 模型 (text_step / frame_step / vocoder_step)...")
        stream_cfg = {}

        if acoustic_model is not None:
            bb = acoustic_model.backbone
            nL = len(bb.layers)
            W = bb.layers[0].window_size
            H = bb.layers[0].n_heads
            dh = bb.layers[0].head_dim
            D = acoustic_model.d_model
            conv_pad = bb.layers[0].conv.dwconv.pad_left
            dur_pads = [dw.pad_left for dw in acoustic_model.dur_predictor.dw_convs]

            ts_wrap = TextStepWrapper(acoustic_model).eval()
            M_demo = bb.chunk_size + bb.lookahead
            ts_inputs = [
                torch.randint(1, len(vocab), (1, M_demo), dtype=torch.long),
                torch.ones((1, M_demo), dtype=torch.long),
                torch.zeros((1, M_demo), dtype=torch.long),
                torch.tensor([bb.chunk_size], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([0], dtype=torch.long),
            ]
            in_names = ["phoneme_ids", "lang_ids", "boundary_ids", "commit_len", "speed", "past_len"]
            out_names = ["h_frame", "durations", "new_past_len"]
            for i in range(nL):
                ts_inputs.append(torch.zeros(1, H, W, dh, dtype=torch.float32))
                in_names.append(f"k_cache_{i}")
                out_names.append(f"new_k_cache_{i}")
            for i in range(nL):
                ts_inputs.append(torch.zeros(1, H, W, dh, dtype=torch.float32))
                in_names.append(f"v_cache_{i}")
                out_names.append(f"new_v_cache_{i}")
            for i in range(nL):
                ts_inputs.append(torch.zeros(1, D, conv_pad, dtype=torch.float32))
                in_names.append(f"conv_cache_{i}")
                out_names.append(f"new_conv_cache_{i}")
            for i, p in enumerate(dur_pads):
                ts_inputs.append(torch.zeros(1, D, p, dtype=torch.float32))
                in_names.append(f"dur_cache_{i}")
                out_names.append(f"new_dur_cache_{i}")

            ts_onnx = out_dir / "text_step.onnx"
            _export_onnx(
                ts_wrap,
                tuple(ts_inputs),
                ts_onnx,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes={
                    "phoneme_ids": {1: "M"},
                    "lang_ids": {1: "M"},
                    "boundary_ids": {1: "M"},
                    "h_frame": {1: "F_chunk"},
                    "durations": {1: "C"},
                },
                opset=args.opset,
                exporter=args.exporter,
            )
            exported_files.append(ts_onnx)

            # frame_step.onnx
            fs_wrap = FrameStepWrapper(acoustic_model, mel_mean, mel_std).eval()
            fp_kl = acoustic_model.frame_pre.k_left
            fp_la = acoustic_model.frame_pre.lookahead
            f0_pads = [dw.pad_left for dw in acoustic_model.f0_uv_predictor.dw_convs]
            en_pads = [dw.pad_left for dw in acoustic_model.energy_predictor.dw_convs]
            dec_pads = [blk.dwconv.pad_left for blk in acoustic_model.frame_decoder]

            fs_inputs = [torch.randn(1, fp_kl + 8 + fp_la, D, dtype=torch.float32)]
            fs_in_names = ["h_frame_win"]
            fs_out_names = ["mel"]
            for prefix, pads in (("f0_cache", f0_pads), ("energy_cache", en_pads), ("fdec_cache", dec_pads)):
                for i, p in enumerate(pads):
                    fs_inputs.append(torch.zeros(1, D, p, dtype=torch.float32))
                    fs_in_names.append(f"{prefix}_{i}")
                    fs_out_names.append(f"new_{prefix}_{i}")

            fs_onnx = out_dir / "frame_step.onnx"
            _export_onnx(
                fs_wrap,
                tuple(fs_inputs),
                fs_onnx,
                input_names=fs_in_names,
                output_names=fs_out_names,
                dynamic_axes={"h_frame_win": {1: "F_win"}, "mel": {2: "F_emit"}},
                opset=args.opset,
                exporter=args.exporter,
            )
            exported_files.append(fs_onnx)

            stream_cfg["acoustic"] = {
                "d_model": D,
                "chunk_size": bb.chunk_size,
                "lookahead": bb.lookahead,
                "n_layers": nL,
                "n_heads": H,
                "head_dim": dh,
                "window_size": W,
                "conv_pad": conv_pad,
                "dur_pads": dur_pads,
                "frame_pre_k_left": fp_kl,
                "frame_pre_lookahead": fp_la,
                "f0_pads": f0_pads,
                "energy_pads": en_pads,
                "fdec_pads": dec_pads,
            }

        if vocoder_model is not None:
            vs_wrap = VocoderStepWrapper(vocoder_model).eval()
            v_dim = vocoder_model.dim
            v_kl = vocoder_model.embed.k_left
            v_la = vocoder_model.embed.lookahead
            hc1_pad = vocoder_model.harm_source.pitch_conv1.pad_left
            hc2_pad = vocoder_model.harm_source.pitch_conv2.pad_left
            blk_pads = [b.dwconv.pad_left for b in vocoder_model.blocks]

            vs_inputs = [
                torch.randn(1, n_mels, v_kl + 8 + v_la, dtype=torch.float32),
                torch.zeros(1, v_dim, hc1_pad, dtype=torch.float32),
                torch.zeros(1, v_dim // 2, hc2_pad, dtype=torch.float32),
                torch.zeros(1, 1, dtype=torch.float64),
                torch.full((1, 1), -1.0, dtype=torch.float64),
            ]
            vs_in_names = ["mel_win", "harm_conv1_cache", "harm_conv2_cache", "phi", "prev_f0"]
            vs_out_names = ["time_frames", "new_harm_conv1_cache", "new_harm_conv2_cache", "new_phi", "new_prev_f0"]
            for i, p in enumerate(blk_pads):
                vs_inputs.append(torch.zeros(1, v_dim, p, dtype=torch.float32))
                vs_in_names.append(f"block_cache_{i}")
                vs_out_names.append(f"new_block_cache_{i}")

            vs_onnx = out_dir / "vocoder_step.onnx"
            _export_onnx(
                vs_wrap,
                tuple(vs_inputs),
                vs_onnx,
                input_names=vs_in_names,
                output_names=vs_out_names,
                dynamic_axes={"mel_win": {2: "M_win"}, "time_frames": {1: "F_emit"}},
                opset=args.opset,
                exporter=args.exporter,
            )
            exported_files.append(vs_onnx)

            stream_cfg["vocoder"] = {
                "dim": v_dim,
                "k_left": v_kl,
                "lookahead": v_la,
                "harm_conv1_pad": hc1_pad,
                "harm_conv2_pad": hc2_pad,
                "block_pads": blk_pads,
                "stream_mel_first_chunk": max(6, v_la + 4),
                "stream_mel_chunk": int(cfg.get("stream_mel_chunk", 16)),
            }

        meta["streaming"] = stream_cfg
        meta["modes"].append("streaming")

    # 写出元数据 JSON（infer_onnx.py 直接读取，无需手传任何超参）
    meta_path = out_dir / "onnx_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Meta] 元数据已写入: {meta_path}")

    # 可选动态 INT8 量化
    if args.quantize_int8:
        print("\n[Stage] 生成权重动态 INT8 量化模型 (*.int8.onnx)...")
        for f in exported_files:
            quantize_onnx_int8(f)

    # 数值等价校验
    if not args.skip_verify:
        verify_exports(out_dir, meta, acoustic_model=acoustic_model, vocoder_model=vocoder_model)

    print(f"\n[Done] ONNX 导出完成，共耗时 {time.perf_counter() - t0:.2f}s，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
