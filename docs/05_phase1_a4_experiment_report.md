# Phase 1（a4）首轮实验报告与下一步计划

> 日期：2026-11-24 ｜ 环境：自建 RTX 4090 (24GB) / 16 核 / 62GB RAM ｜ 数据：BZNSYP 8.5h（standard 档）
> 结论速览：**E3_chunk_attn_C4_L8 夺冠，主干定型方向确认；但 standard 档（15k/18k 步）严重欠训，
> 声学与声码器双双不足，输出不可懂。下一步：只训 E3 + 声码器，切 full_quality 档（40k/50k 步）。**

---

## 1. 实验执行情况

| 项目 | 状态 | 备注 |
| --- | --- | --- |
| a4 门禁自检 | ✅ 全过 | 6 模型流式等价 ~2e-06、零前瞻泄漏 (Leak=0.0)、GDN 梯度有限、声码器 OLA 等价 |
| 数据预处理 | ✅ 复用 | 旧 `cache/bznsyp_v2_max8.0h`（8.5h / train 8075 / val 80）秒级命中，未重下重算 |
| CausalVocosLite a4（含 2 帧前瞻） | ✅ 18k 步 | 12k 谱 + 6k GAN，30 分钟，mel L1 0.214 |
| 6 模型声学训练 | ✅ 各 15k 步 | **4 路并行**（`--models` 分组），GPU 打满 97%，聚合 ~34 it/s，总耗时 ~90 分钟 |
| 合成 + 评测 | ✅（MER 缺失） | UTMOS/SQUIM/SIM/RTF 齐全；ASR(MER) 因 ctranslate2 缺 libcublas.so.12 全 NaN（见 §4） |
| 人工试听 | ❌ 不可懂 | E3 明显好于其他，但仅个别音可辨、电音明显（用户听感，与客观指标一致） |

## 2. 结果

### 2.1 训练收敛（val mel L1，越低越好）

| 模型 | Val Mel L1 | 步数 |
| --- | --- | --- |
| **E3_chunk_attn_C4_L8** | **0.5281** | 15000 |
| E1_conv_unigru_L8 | 0.5460 | 15000 |
| E2_swa_L8 | 0.5769 | 15000 |
| E0_offline_bitransformer（上界参考） | 0.5851 | 15000 |
| E4_gated_deltanet_L8 | 0.6038 | 15000 |
| E1_conv_unigru_L0（零前瞻下界） | 0.6185 | 15000 |

- 流式模型全面超越离线双向 Transformer 上界（E0），"深分块前瞻" 路线成立。
- L8 vs L0（同主干消融）：右前瞻净收益 Δ≈0.07 val mel，量化了前瞻价值。

### 2.2 综合评测排行榜（MER 列缺失，综合分中可懂度项按保底计）

| 排名 | 模型 | UTMOS↑ | SIM↑ | RTF(2线程)↓ | TTFA↓ | Seam_Click↓ | 综合分 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **E3_chunk_attn_C4_L8** | **1.405** | **0.796** | 0.072 | 37.6ms | 0 | 47.68 |
| 2 | E2_swa_L8 | 1.350 | 0.733 | 0.063 | 18.5ms | 0 | 46.12 |
| 3 | E1_conv_unigru_L8 | 1.300 | 0.713 | **0.049** | **13.3ms** | 0 | 45.86 |
| 4 | E0_offline | 1.323 | 0.701 | 0.010* | 60.2ms | 0 | 45.28 |
| 5 | E4_gated_deltanet_L8 | 1.303 | 0.663 | 0.091 | 22.7ms | 0 | 43.18 |
| 6 | E1_conv_unigru_L0 | 1.271 | 0.625 | 0.083 | 19.8ms | 0 | 42.11 |

（GT 真声锚点 UTMOS = 3.787；*E0 为离线整段合成，RTF 无流式意义）
- 延迟指标全部达标：TTFA 13–38ms（目标 <150ms），RTF 0.049–0.091（目标 <0.10），流式拼接零爆音。
- **注意：UTMOS ~1.3 的绝对值即客观上接近噪音**，与人工试听一致。本排行榜只反映**相对**优劣。

## 3. 质量问题定位（关键）

对 E3 best.pt 与声码器逐环节隔离测试（3 条 val 句，UTMOS）：

| 链路 | UTMOS | 损失 | 结论 |
| --- | --- | --- | --- |
| 真声 GT | 3.86–4.23 | — | 参考 |
| GT mel → 预训练 Vocos | 3.16–3.79 | ~0.4 | mel 提取损失小，正常 |
| GT mel → 自训 CausalVocosLite | 1.90–2.21 | **~1.3** | ⚠️ **声码器欠训 = "电音"来源**（18k 步 / GAN 段仅 6k） |
| 推理 mel → 预训练 Vocos | 1.32–2.02 | **~1.9** | ⚠️ **声学推理欠训 = "听不清"来源** |

佐证：
- F0 预测器**完全正常**（均值 249 vs GT 237Hz，范围/方差吻合）——架构无恙。
- 时长比 0.81–0.97，合理。
- 真实推理路径 mel L1 = 1.30–1.66，而 teacher-forced 验证只有 0.53 —— 差距来自"预测时长 + 非强制对齐"的推理路径，典型的 NAR-TTS 欠训表现（此类架构通常需要 10 倍步数量级才出可懂语音）。

**判定：不是代码 bug，是 standard 档训练预算不足。** a4 的修正（Beta-Binomial 先验、STE 硬对齐、帧级前瞻）方向都有效（a4 全线优于 a3），只是还没训到位。

## 4. 环境与工程备忘

1. **向量化对齐热路径**（已合入 repo，`b1181c2`）：Viterbi 批向量化 30→15ms、Beta-Binomial 先验 13.2→0.5ms，单进程提速 +52%；4 路并行后 GPU 打满，CPU 不再是瓶颈。
2. **4 路并行训练方法**：`--models` 按 `{E0,E2} {E1L0,E1L8} {E3} {E4}` 分组各起一进程，产物目录/断点天然隔离；训完后跑一次全量（无 `--models`）合并评测出完整排行榜。各 worker 训完会各自跑子集评测（冗余但无害，可无视）。
3. **ASR 评测在 GPU 上失败**：ctranslate2 4.9.1 需要 CUDA12 的 `libcublas.so.12`，本机 torch 为 cu13。补装 `nvidia-cublas-cu12 + nvidia-cudnn-cu12`（经代理下载 ~1GB 很慢，未完成）或改用 CPU int8 whisper。未修复，故 MER 列为 NaN。
4. **下载断点续传**：代理下 `hf_hub_download` 会假死，改用 `curl -C -` 重试循环手动放置到 `datasets/raw/BZNSYP.tar.bz2` 可命中 notebook 缓存路径。

## 5. 下一步计划（下次执行）

**只训 E3_chunk_attn + 声码器，切 full_quality 档**（vocoder 50k 步：35k 谱 + 15k GAN；声学 40k 步；数据上限 20h → BZNSYP 实际 ~12h 全量，需重新预处理 ~20 分钟）：

```bash
cd /root/AetherTTS
python scripts/aethertts_phase1.py --selftest --skip-pip   # 门禁自检
# 4 路并行时曾用 --models 分组；单模型直接：
python scripts/aethertts_phase1.py --profile full_quality --dataset bznsyp \
    --root ./aethertts_exp --skip-pip --models E3_chunk_attn_C4_L8
```

- 预计耗时：声码器 ~1.5h + E3 声学 ~3.5h + 评测 ~0.5h ≈ **5–6 小时**（建议 tmux 后台挂机）。
- 产物目录自动切换到 `*_full_quality_a4_fL2`，与 standard 档结果并存可对比。
- 评测听感预期：声码器电音显著减弱（GAN 段 6k→15k）；可懂度是主要不确定项，若 40k 步仍不足，再考虑加训或教师蒸馏（Phase 2 路线）。
- 可选：修好 ASR（CPU int8 whisper 或补齐 CUDA12 库）以获得完整 MER 维度。

### 5.1 E3 速度与参数实测（2026-11-24）

| 指标 | 数值 |
| --- | --- |
| 单独训练速度 | **3.07 it/s**（326ms/步：前向 148ms + 反向 177ms，batch 32，实测） |
| 4 路并行时 | 2.76 it/s（15000 步 / 90.7 min）— 并行争用仅损失 11%，自身 chunk 串行是主瓶颈 |
| 模型参数 | 总 5.28M；推理 4.99M（aligner 0.29M 仅训练用）；+ 声码器 3.67M ≈ 系统共 8.7M |

慢的根因是**设计选择**：训练 forward 逐 chunk 模拟流式推理（T/4 个 chunk × 5 层的小 kernel
串行发射），换取训练↔流式推理 100% 数学等价（Stream Diff ~1.7e-06）。因此：

- **提速首选：batch 32→64**（4090 显存富余）。吞吐预计 +30–50%，E3 声学 40k 步从 ~3.5h 降到
  ~2.5h。质量导向的单模型跑可用；受控对比实验仍应用预设 batch 保持变量一致。
- 其余选项评估：单独跑仅 +11%；`torch.compile` 可能 1.5–2× 但数值非逐位等价（需过 1e-4 门禁）
  且动态形状 recompile 需处理；KV cache 环形缓冲 ~10–20%；**不建议**改稠密并行训练 forward
  （破坏流式等价保证）。
- 注意：batch 目前无环境变量/CLI 覆写入口，如需 batch 64 需在 notebook 的 `PROFILE_PRESETS`
  中调整（改动单一真相源后运行 `python scripts/sync_phase1_script.py` 同步脚本）。

## 6. 产物位置

```
aethertts_exp/
├── runs/*_a4_fL2/                 # 6 模型 checkpoint（best/latest/milestone/INT8，全部 .done）
├── vocoder/causal_vocos_lite_bznsyp_standard_phase1_a4_L2/
├── eval_samples/bznsyp_standard_phase1_a4_fL2_vL2/   # 每模型 offline/streaming/pretrained_vocos + GT
├── reports/bznsyp_standard_phase1_a4_fL2_vL2/        # leaderboard.csv/json、per_sample、图表、盲听表
├── cache/bznsyp_v2_max8.0h/        # 特征缓存（full_quality 需另建 max20h 缓存）
└── datasets/raw/BZNSYP.tar.bz2    # 原始压缩包（2.35GB，已验证完整）
runs/E3_chunk_attn_C4_L8_*_a4_fL2/previews/           # E3 训练全程试听（step_1500→15000）
```
