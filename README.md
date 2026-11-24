# AetherTTS

面向 **嵌入式 Linux（中低端手机 SoC 水平、纯 CPU）** 的高自然度中英双语（支持中英混读）**双向流式**语音合成系统，目标听感对标 Siri / 小爱同学。

> **当前阶段：Phase 0 — 调研、架构蓝图与训练数据整理**
> 本阶段聚焦于模型技术路线锁定、带右前瞻的流式输入主干调研，以及商用合规的中英双语/混读训练数据收集与蒸馏方案设计。端侧 C++ 推理引擎将在模型实验收敛后再启动开发。

---

## 核心设计目标与约束

| 维度 | 目标指标 / 选型决策 |
| :--- | :--- |
| **运行硬件** | 嵌入式 Linux（ARMv8 Cortex-A55/A76 级别 SoC），**纯 CPU**（预算 ≤ 2 大核，RAM < 150 MB） |
| **音质与音色** | **单音色特化**（对标 Siri / 小爱同学，零幻觉/零吞字、准确多音字与韵律、24 kHz 清晰音质） |
| **语种能力** | 中文 + 英文 + 偶尔中英混读（Code-Switching） |
| **生成范式** | **魔改 Kokoro 单步直出路线**（~8M–12M，单音色移除 Speaker Embedding，显式 Duration/F0/Energy + 瘦身因果 iSTFTNet/Vocos + 训练期对抗/教师蒸馏） |
| **流式能力** | **流式输入（对接 LLM）+ 流式输出**，必须具备**有限右前瞻（Finite Right Lookahead）**以保证中文变调/多音字/韵律自然度 |
| **延迟与速度** | 目标纯 CPU `RTF < 0.10`，TTS 自身首包延迟 `TTFA < 150 ms` |
| **数据与训练控时** | **严格控量（15–25 小时黄金单音色语料）**，商用友好优先 + Apache 2.0 大模型离线蒸馏造数，确保单卡训练时间可控（小时至 1 天级收敛） |

---

## 仓库目录与文档导航

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suulnnka/AetherTTS/blob/main/notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb) — 在 Google Colab 打开（自动云盘高频断点备份）；自建 RTX 4090 / Linux 服务器可直接打开同一 Notebook 或用 `scripts/run_phase1_local.py` 后台运行（全部文件纯本地保存）。

```bash
# 自建 RTX 4090 服务器推荐用法（纯本地保存，配合 tmux 离线挂机）：
python scripts/run_phase1_local.py --profile standard_phase1 --dataset bznsyp --root ./aethertts_exp
```

```text
AetherTTS/
├── README.md                                    # 项目总览与路线图（当前文件）
├── notebooks/
│   └── AetherTTS_Colab_Phase1_Experiment.ipynb  # Phase 1 六主干流式实验（Colab 云盘高频备份 + 自建 4090 纯本地双兼容）
├── scripts/
│   └── run_phase1_local.py                      # 自建 Linux GPU / RTX 4090 命令行无界面启动器（直接执行上述 Notebook）
├── data/
│   ├── datasets.yaml                            # 训练/前端/评测/蒸馏教师资源结构化清单（含已核实链接与协议）
│   └── README.md                                # 严控体量的单音色数据策略与商用许可分级
└── docs/
    ├── 01_architecture_overview.md              # AetherTTS 8M-12M 单音色单步流式整体架构与算力预算
    ├── 02_streaming_lookahead_survey.md         # 流式输入 + 有限右前瞻深度调研（SWA / Mamba-SSM / 线性注意力 / 门控 DeltaNet 对比与实验矩阵）
    ├── 03_teacher_distillation_pipeline.md      # 商用友好大模型教师离线造数与两段式特征/对抗蒸馏方案
    └── 04_evaluation_protocol.md                # 四维客观自动评测 + 50 句黄金人工盲听质量评价体系
```

---

## 项目路线图（Roadmap）

- [x] **Phase 0：技术调研与数据清单**
  - 锁定「10M–15M 魔改 Kokoro 单步直出 + 有限右前瞻流式」总体方向
  - 整理并核实商用友好与学术对照中英双语/混读数据集地址（[`data/datasets.yaml`](data/datasets.yaml)）
  - 制定流式输入主干（SWA / Mamba-SSM / 线性注意力 / 门控 DeltaNet / Conv+GRU 基线）控制变量实验矩阵（[`docs/02_streaming_lookahead_survey.md`](docs/02_streaming_lookahead_survey.md)）
- [ ] **Phase 1：数据管线与主干对比实验（下一步）**
  - 实现统一中英混读前端（TN + G2P + 轻量多音字/韵律）与 24 kHz 预处理/强制对齐管线
  - 在相同右前瞻窗口下，对比验证 SWA、Mamba/SSM、线性注意力、门控 DeltaNet 与因果卷积基线的音质、边界一致性与 CPU 速度
  - 选出综合最优的流式主干定型
- [ ] **Phase 2：双语精品音色训练、教师蒸馏与量化验证**
  - 执行教师模型离线造数（补齐单音色中英混读短板）与训练期 SLM/MPD 对抗训练
  - 验证 FP32 / FP16 / INT8 混合精度量化及流式状态缓存（Zero-Click）的数值一致性
- [ ] **Phase 3：嵌入式 Linux C++ 纯 CPU 流式推理引擎**
  - 基于定型模型导出带显式状态缓存的 ONNX / MNN 计算图
  - 实现嵌入式 Linux C++17 流式前端 + 推理调度 + ALSA/PCM 低延迟流式输出与 ARM64 实测调优
