# AetherTTS

面向 **嵌入式 Linux（中低端手机 SoC 水平、纯 CPU）** 的高自然度中英双语（支持中英混读）**双向流式**语音合成系统，目标听感对标 Siri / 小爱同学。

> **当前阶段：Phase 1 收官与闭环质量冲刺 — 锁定唯一主力 `E3_chunk_attn_C4_L8`（`a5` 架构）**
> 经首轮 6 槽控制变量实验验证，**5 层深双向分块前瞻（`ChunkLookaheadBackbone`, `C=4, L=8, W=48`）** 全面胜出并正式定型为唯一默认路线（其余 `E0/E1/E2/E4` 对照代码归档保留，默认不再运行）。`a5` 版本进一步完成**完备中英音系前端（`v3`）、浊音感知 `F0+UV`、2D 共振峰结构损失、局部窗口相对 RoPE（永久 $O(1)$ 长流式）、帧率谐波源因果声码器（`CausalVocosLite+Harm`）+ 转置 `SpecDisc` 修复、以及「声学 → GTA 预测谱 → 声码器」闭环精调**。

---

## 核心设计目标与约束

| 维度 | 目标指标 / 选型决策 |
| :--- | :--- |
| **运行硬件** | 嵌入式 Linux（ARMv8 Cortex-A55/A76 级别 SoC），**纯 CPU**（预算 ≤ 2 大核，RAM < 150 MB） |
| **音质与音色** | **单音色特化**（对标 Siri / 小爱同学，零幻觉/零吞字、准确多音字与韵律、24 kHz 清晰音质） |
| **语种能力** | 中文 + 英文 + 偶尔中英混读（Code-Switching） |
| **生成范式** | **单步直出 + GTA 谐波源闭环路线**（推理端到端 `~8.77M`，剥离 Aligner；显式 Duration + 浊音感知 `F0/UV` + Energy + 音素内相位 + 2D 共振峰结构损失 + 帧率包裹相位谐波源 `CausalVocosLite`） |
| **流式能力** | **流式输入（对接 LLM）+ 流式输出**，前端消歧滑窗与声学右前瞻共享 `L=8` 音素预算，未旋转 K 局部相对 RoPE 支持数小时不重置状态，首包 6 帧（64 ms）动态低延迟出声 |
| **延迟与速度** | 目标纯 CPU `RTF < 0.10`，TTS 自身计算首包延迟 `TTFA < 150 ms`（实测 2 线程 CPU ~30–40 ms） |
| **数据与训练控时** | **严格控量（12–25 小时黄金单音色语料）**，商用友好优先 + Apache 2.0 大模型离线蒸馏造数，确保单卡数小时内高质量收敛 |

---

## 仓库目录与文档导航

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suulnnka/AetherTTS/blob/main/notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb) — 在 Google Colab 打开（自动云盘高频断点备份）；自建 RTX 4090 / Linux 服务器可直接打开同一 Notebook，或使用 [`aethertts_phase1.py`](file:///home/antigravity/suulnnka/AetherTTS/scripts/aethertts_phase1.py) 后台运行（全部文件纯本地保存）。

```bash
# 自建 RTX 4090 服务器推荐用法（先跑几秒门禁自检，再 tmux 挂后台跑唯一主力 E3 full_quality）：
python scripts/aethertts_phase1.py --selftest --skip-pip
python scripts/aethertts_phase1.py --profile full_quality --dataset bznsyp --batch-size 64 --root ./aethertts_exp --skip-pip
```

```text
AetherTTS/
├── README.md                                    # 项目总览与四阶段路线图（当前文件）
├── notebooks/
│   └── AetherTTS_Colab_Phase1_Experiment.ipynb  # 单一真相源 Notebook（a5 唯一主力 E3 + GTA 谐波源闭环训练）
├── scripts/
│   ├── aethertts_phase1.py                      # 独立 Python CLI 运行器（含 --selftest / --profile-steps，由 sync 脚本确定性生成）
│   ├── aethertts_lib.py                         # 无副作用纯模型与音系前端库（由 sync 脚本确定性生成，供诊断/推理直接 import）
│   ├── diagnose_phase1.py                       # 8 链路逐级隔离诊断、对齐取证与 ChunkAttn 提速校验脚本
│   ├── run_phase1_local.py                      # Notebook 顺序执行包装器
│   └── sync_phase1_script.py                    # 单一真相源同步校验脚本（Notebook -> aethertts_lib.py + aethertts_phase1.py）
├── data/
│   ├── datasets.yaml                            # 训练/前端/评测/蒸馏教师资源结构化清单（含已核实链接与协议）
│   └── README.md                                # 严控体量的单音色数据策略与商用许可分级
└── docs/
    ├── 01_architecture_overview.md              # AetherTTS 8.8M 单音色流式整体架构、谐波源声码器与 GTA 闭环设计
    ├── 02_streaming_lookahead_survey.md         # 流式输入与有限右前瞻调研（ChunkLookaheadBackbone 深度设计与定型结论）
    ├── 03_teacher_distillation_pipeline.md      # 商用友好大模型教师离线造数与两段式特征/对抗蒸馏方案
    ├── 04_evaluation_protocol.md                # 四维客观自动评测 + 50 句黄金人工盲听质量评价体系
    └── 05_phase1_a4_experiment_report.md        # Phase 1 首轮实验报告、根因审计与 a5 全量修复纪要
```

---

## 项目路线图（Roadmap）

- [x] **Phase 0：技术调研与数据清单**
  - 锁定「单步 NAR + 有限右前瞻流式 + 轻量因果 iSTFT 声码器」总体方向
  - 整理并核实商用友好与学术对照中英双语/混读数据集地址（[`data/datasets.yaml`](data/datasets.yaml)）
- [ ] **Phase 1：流式输入核心架构定型与 `a5` 闭环验证（当前收官阶段）**
  - 完成 6 槽对比实验，**正式锁定 `E3_chunk_attn_C4_L8`（5 层深双向分块前瞻 `ChunkLookaheadBackbone`）为唯一主力主干**（其余对照代码归档保留）
  - 落地 `a5` 全链路修复：完备中英音系与韵律前端（`v3`，20,901 汉字零 UNK + 儿化/变调/词界 `boundary_ids`）、浊音感知 `F0+UV`、2D 共振峰结构损失、局部窗口相对 RoPE（超长流式永久 $O(1)$）、修复转置 `SpecDisc` + `PeriodDisc(11)` + 帧率包裹相位谐波源 `CausalVocosLite`、以及「先声学 → 导出 GTA Mel → 声码器闭环精调」与首包 6 帧动态低延迟流式调度
  - 在 `BZNSYP` 全量 12h 上跑完 `full_quality` 档闭环训练与客观/人工验收
- [ ] **Phase 2：真·流式前端闭环与双语精品音色**
  - 升级为**增量流式双语前端**（2–4 汉字共享滑窗，处理数字 TN / 多音字 / 三声变调 / 英文整词），向 `ChunkLookaheadBackbone.forward_committed_with_lookahead` 输出 `(committed_ids, tentative_lookahead_ids)`
  - 使用 15–25h 单音色精品双语语料 + Apache 2.0 教师离线补齐中英混读短板
  - 完成「多句连续流式输入不重置状态」跨句韵律与零接缝爆音验收
- [ ] **Phase 3：量化验证与三段式流式图导出**
  - 权重 INT8 动态/静态量化（KV/卷积状态、谐波积分相位与 OLA 保持 FP32/FP64），验证 FP32 ↔ INT8 听感无损、内存 `< 80 MB`
  - 固化三段解耦流式接口（`text_step` → `frame_step` → `vocoder_step`）并导出 ONNX 校验数值一致性
- [ ] **Phase 4：嵌入式 Linux C++17 纯 CPU 流式推理引擎**
  - C++17 流式前端 + ONNX Runtime / MNN 纯 CPU 调度 + ALSA/PCM 低延迟环形缓冲，在双核 ARM64 开发板实测达成 `RTF < 0.10`、`TTFA < 150 ms`
