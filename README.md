# AetherTTS

面向 **嵌入式 Linux（中低端手机 SoC 水平、纯 CPU）** 的高自然度中英双语（支持中英混读）**双向流式**语音合成系统，目标听感对标 Siri / 小爱同学。

> **当前阶段：Phase 1 — 聚焦「流式输入」的主干与前瞻机制对比定型（`a4` 架构）**
> 一切设计以 **LLM 逐字流式输入 → 嵌入式纯 CPU 实时出声** 为核心约束：对比零前瞻下界、浅层集中前瞻、**多层深双向分块前瞻（Chunk-wise Lookahead Attention）** 与标准门控 DeltaNet，锁定唯一主干后再进入流式前端闭环、精品音色与 C++ 引擎开发。

---

## 核心设计目标与约束

| 维度 | 目标指标 / 选型决策 |
| :--- | :--- |
| **运行硬件** | 嵌入式 Linux（ARMv8 Cortex-A55/A76 级别 SoC），**纯 CPU**（预算 ≤ 2 大核，RAM < 150 MB） |
| **音质与音色** | **单音色特化**（对标 Siri / 小爱同学，零幻觉/零吞字、准确多音字与韵律、24 kHz 清晰音质） |
| **语种能力** | 中文 + 英文 + 偶尔中英混读（Code-Switching） |
| **生成范式** | **魔改 Kokoro 单步直出路线**（~8.5M–10M，单音色移除 Speaker Embedding，显式 Duration/F0/Energy + 音素内相位 + 有限前瞻帧解码器/因果 Vocos-Lite + 训练期对抗/教师蒸馏） |
| **流式能力** | **流式输入（对接 LLM）+ 流式输出**，前端消歧滑窗与声学右前瞻共享 4–8 音素（2–4 汉字）预算，全链路零前瞻膨胀 |
| **延迟与速度** | 目标纯 CPU `RTF < 0.10`，TTS 自身首包延迟 `TTFA < 150 ms`（当前设计预算 ~85–100 ms） |
| **数据与训练控时** | **严格控量（15–25 小时黄金单音色语料）**，商用友好优先 + Apache 2.0 大模型离线蒸馏造数，确保单卡训练时间可控（小时至 1 天级收敛） |

---

## 仓库目录与文档导航

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suulnnka/AetherTTS/blob/main/notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb) — 在 Google Colab 打开（自动云盘高频断点备份）；自建 RTX 4090 / Linux 服务器可直接打开同一 Notebook，或使用 `scripts/aethertts_phase1.py` / `scripts/run_phase1_local.py` 后台运行（全部文件纯本地保存）。

```bash
# 自建 RTX 4090 服务器推荐用法（先跑几秒门禁自检，再 tmux 挂后台训练）：
python scripts/aethertts_phase1.py --selftest --skip-pip
python scripts/aethertts_phase1.py --profile standard_phase1 --dataset bznsyp --root ./aethertts_exp
```

```text
AetherTTS/
├── README.md                                    # 项目总览与四阶段路线图（当前文件）
├── notebooks/
│   └── AetherTTS_Colab_Phase1_Experiment.ipynb  # Phase 1 聚焦流式输入的 6 槽对比实验（Colab 云盘备份 + 自建 4090 纯本地双兼容）
├── scripts/
│   ├── run_phase1_local.py                      # 自建 Linux GPU / RTX 4090 命令行启动器（直接执行上述 Notebook）
│   ├── aethertts_phase1.py                      # 独立 Python 运行器（含 --selftest / --profile-steps，由 sync 脚本从 Notebook 1:1 同步）
│   └── sync_phase1_script.py                    # 单一真相源同步脚本：从 Notebook 确定性生成并校验 scripts/aethertts_phase1.py
├── data/
│   ├── datasets.yaml                            # 训练/前端/评测/蒸馏教师资源结构化清单（含已核实链接与协议）
│   └── README.md                                # 严控体量的单音色数据策略与商用许可分级
└── docs/
    ├── 01_architecture_overview.md              # AetherTTS 8M-10M 单音色单步流式整体架构、延迟与算力预算
    ├── 02_streaming_lookahead_survey.md         # 流式输入与有限右前瞻调研（含 Chunk-wise Lookahead Attention 深度设计与 Phase 1 矩阵）
    ├── 03_teacher_distillation_pipeline.md      # 商用友好大模型教师离线造数与两段式特征/对抗蒸馏方案
    └── 04_evaluation_protocol.md                # 四维客观自动评测 + 50 句黄金人工盲听质量评价体系
```

---

## 项目路线图（Roadmap）

- [x] **Phase 0：技术调研与数据清单**
  - 锁定「单步 NAR + 有限右前瞻流式 + 轻量因果 iSTFT 声码器」总体方向
  - 整理并核实商用友好与学术对照中英双语/混读数据集地址（[`data/datasets.yaml`](data/datasets.yaml)）
- [ ] **Phase 1：流式输入核心架构定型（当前阶段，`a4`）**
  - 精简冗余算子（砍掉短句冗余的 SSM / 累加 WKV），聚焦 6 槽流式输入矩阵：`E0` 离线天花板、`E1`（`L=0` vs `L=8` 前瞻拐点）、`E2`（SWA 浅前瞻）、**`E3`（5 层深双向分块前瞻 `ChunkLookaheadBackbone`, `C=4, L=8`）**、`E4`（标准门控 DeltaNet $O(1)$ 独苗）
  - 补齐 `frame_decoder` 2 帧集中式右前瞻 + 音素内多尺度相位特征 + 声码器 2 帧右前瞻，通过零前瞻膨胀与流式↔离线数学等价硬门禁
  - 在 BZNSYP / LJSpeech 上完成四维评测与盲听，**锁定唯一主干定型**
- [ ] **Phase 2：真·流式前端闭环与双语精品音色**
  - 升级为**增量流式双语前端**（2–4 汉字共享滑窗，处理数字 TN / 多音字 / 三声变调 / 英文整词），向声学主干输出 `(committed_ids, tentative_lookahead_ids)`，不串行叠加首包延迟且支持暂定纠错
  - 使用 15–25h 单音色精品双语语料 + Apache 2.0 教师离线补齐中英混读短板 + 训练期对抗损失精调
  - 增加「多句连续流式输入不重置状态」专项评测（跨句韵律、停顿与零接缝爆音）
- [ ] **Phase 3：量化验证与三段式流式图导出**
  - 权重 INT8 动态/静态量化（KV/递推状态与 OLA 保持 FP32），验证 FP32 ↔ INT8 听感无损、内存 `< 80 MB`
  - 固化三段解耦流式接口（`text_step` → `frame_step` → `vocoder_step`）并导出 ONNX 校验数值一致性
- [ ] **Phase 4：嵌入式 Linux C++17 纯 CPU 流式推理引擎**
  - C++17 流式前端 + ONNX Runtime / MNN 纯 CPU 调度 + ALSA/PCM 低延迟环形缓冲，在双核 ARM64 开发板实测达成 `RTF < 0.10`、`TTFA < 150 ms`
