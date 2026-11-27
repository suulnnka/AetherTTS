# AetherTTS

面向 **嵌入式 Linux（中低端手机 SoC 水平、纯 CPU）** 的高自然度中英双语（支持中英混读）**双向流式**语音合成系统，目标听感对标 Siri / 小爱同学。

> **当前阶段：Phase 1 音质精进与端侧 INT8 闭环 — 主力 `E3_chunk_attn_C4_L8`（`a10` 架构 / `v5` 特征缓存）**
> 经 `a8` 解决对齐坍缩、`a9` 实机 2 线程 INT8 流式达成 `TTFA ≈ 27.8 ms` / `RTF ≈ 0.149` 并消除强电音与 OLA 空洞后，`a10` 基于 `a9` 10 句标贝真声 5 路隔离 + ONNX 内部探针取证，针对残留的「略微断层电音」与「声音发闷」完成全链路升级：**`v5` 真清浊音特征（FFT-YIN CMNDF + NACF + 频谱高低频比，将清辅音误判 `uv=1` 从 99.7% 压至 0%）+ 声学训练期推理零开销双尺度 2D `MelPatchDiscriminator`（+ Feature Matching + 1–12 kHz 频带加权 + 解码器末段 Dropout 退火）+ `CausalVocosLite` 48 阶平滑奈奎斯特与因果线性啁啾谐波源（四声滑音 OLA 一致性 `0.67 → 0.999`）+ 滤波相位平滑/浊音低频噪声抑制正则 + `per-channel` 与离群保护 INT8 导出**（详见 [`docs/07_phase1_a9_forensics_and_a10_upgrade.md`](docs/07_phase1_a9_forensics_and_a10_upgrade.md)）。

---

## 核心设计目标与约束

| 维度 | 目标指标 / 选型决策 |
| :--- | :--- |
| **运行硬件** | 嵌入式 Linux（ARMv8 Cortex-A55/A76 级别 SoC），**纯 CPU**（预算 ≤ 2 大核，RAM < 150 MB） |
| **音质与音色** | **单音色特化**（对标 Siri / 小爱同学，零幻觉/零吞字、准确多音字与韵律、24 kHz 清晰音质） |
| **语种能力** | 中文 + 英文 + 偶尔中英混读（Code-Switching） |
| **生成范式** | **单步直出 + 连续相位啁啾谐波源-滤波器闭环路线**（推理端到端 `~11.50M` = 声学 `5.15M` + 声码器 `6.35M`，剥离 Aligner/深监督头/训练期 2D Mel 判别器；显式 Duration + 共享前瞻多尺度空洞 `F0/UV`/Energy + 6 层空洞帧解码器 + 48 阶切比雪夫因果线性啁啾 STFT 谐波源 `CausalVocosLite`） |
| **流式能力** | **流式输入（对接 LLM）+ 流式输出**，前端消歧滑窗与声学右前瞻共享 `L=8` 音素预算，未旋转 K 局部相对 RoPE 支持数小时不重置状态，首包 `8` 帧（85 ms，匹配 4 帧声码器前瞻且仍在首个文本块立即出声）动态低延迟吐出 |
| **延迟与速度** | 目标纯 CPU `RTF < 0.10`，TTS 自身计算首包延迟 `TTFA < 150 ms`（桌面 2 线程 INT8 已实测 `TTFA ≈ 28 ms`、`RTF ≈ 0.149`） |
| **数据与训练控时** | **严格控量（12–25 小时黄金单音色语料）**，商用友好优先 + Apache 2.0 大模型离线蒸馏造数，确保单卡数小时内高质量收敛 |

---

## 仓库目录与文档导航

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suulnnka/AetherTTS/blob/main/notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb) — 在 Google Colab 打开（自动云盘高频断点备份）；自建 RTX 4090 / Linux 服务器可直接打开同一 Notebook，或使用 [`aethertts_phase1.py`](scripts/aethertts_phase1.py) 后台运行（全部文件纯本地保存）。

```bash
# 自建 RTX 4090 服务器推荐用法（先跑几秒门禁自检，再 tmux 挂后台跑唯一主力 E3 full_quality）：
python scripts/aethertts_phase1.py --selftest --skip-pip
python scripts/aethertts_phase1.py --profile full_quality --dataset bznsyp --batch-size 64 --root ./aethertts_exp --skip-pip

# 训练完成后导出离线 + 三段式流式 ONNX（默认 balanced INT8：QUInt8 per-channel 防 AVX2 饱和溢出 + 保护声码器高离群 pw2 层）：
python scripts/export_onnx.py --root ./aethertts_exp --profile full_quality --out-dir ./onnx_export --quantize-int8
```

```text
AetherTTS/
├── README.md                                     # 项目总览与四阶段路线图（当前文件）
├── notebooks/
│   └── AetherTTS_Colab_Phase1_Experiment.ipynb   # 单一真相源 Notebook（a10 唯一主力 E3 + v5 真清浊音 + MelGAN + 48 阶啁啾谐波源）
├── scripts/
│   ├── aethertts_phase1.py                       # 独立 Python CLI 运行器（含 --selftest / --profile-steps，由 sync 脚本确定性生成）
│   ├── aethertts_lib.py                          # 无副作用纯模型与音系前端库（由 sync 脚本确定性生成，供诊断/推理直接 import）
│   ├── export_onnx.py                            # 离线两段式 + 三段式流式 ONNX 导出、per-channel INT8 量化与数值一致性校验
│   ├── infer_onnx.py                             # 零 PyTorch 依赖纯 CPU ONNX Runtime 推理脚本
│   ├── aethertts_server_onnx.py                  # ONNX Runtime 流式 HTTP 演示服务 + web/index.html
│   ├── diagnose_phase1.py                        # 8 链路逐级隔离诊断、对齐取证、93.75 Hz 调制/空洞检测与 ChunkAttn 校验
│   ├── run_phase1_local.py                       # Notebook 顺序执行包装器
│   └── sync_phase1_script.py                     # 单一真相源同步校验脚本（Notebook -> aethertts_lib.py + aethertts_phase1.py）
├── data/
│   ├── datasets.yaml                             # 训练/前端/评测/蒸馏教师资源结构化清单（含已核实链接与协议）
│   └── README.md                                 # 严控体量的单音色数据策略与商用许可分级
└── docs/
    ├── 01_architecture_overview.md               # AetherTTS 11.5M 单音色流式整体架构、谐波源声码器与 GTA 闭环设计
    ├── 02_streaming_lookahead_survey.md          # 流式输入与有限右前瞻调研（ChunkLookaheadBackbone 深度设计与定型结论）
    ├── 03_teacher_distillation_pipeline.md       # 商用友好大模型教师离线造数与两段式特征/对抗蒸馏方案
    ├── 04_evaluation_protocol.md                 # 四维客观自动评测（含 Mod_93Hz / Energy_Holes / Formant_Sharp）+ 50 句人工盲听体系
    ├── 05_phase1_a4_experiment_report.md         # Phase 1 首轮实验报告、根因审计与 a5 修复纪要
    ├── 06_phase1_a8_forensics_and_a9_upgrade.md  # Phase 1 a8 61 句实机取证与 a9 全链路升级
    └── 07_phase1_a9_forensics_and_a10_upgrade.md # Phase 1 a9 INT8 流式取证（发闷/断层电音/离群量化）与 a10 升级方案
```

---

## 项目路线图（Roadmap）

- [x] **Phase 0：技术调研与数据清单**
  - 锁定「单步 NAR + 有限右前瞻流式 + 轻量因果 iSTFT 声码器」总体方向
  - 整理并核实商用友好与学术对照中英双语/混读数据集地址（[`data/datasets.yaml`](data/datasets.yaml)）
- [ ] **Phase 1：流式输入核心架构定型与 `a10` 音质精进闭环（当前收官阶段）**
  - 完成 6 槽对比实验锁定 `E3_chunk_attn_C4_L8`；完成 `a6–a9` 完备音系前端、规范 `blank=-1.0` Forward-Sum 防坍缩对齐、连续相位谐波源声码器与三段式流式 ONNX/INT8 端到端链路
  - 基于 `a9` INT8 流式实机 5 路隔离取证落地 **`a10`（`v5` 特征缓存）升级**：FFT-YIN CMNDF + NACF + 频谱高低频比真清浊音特征（根治 99.7% 清辅音错标 `uv=1`）、声学训练期双尺度 2D `MelPatchDiscriminator`（推理零开销，消除纯 L1 发闷）、`CausalVocosLite` 48 阶平滑奈奎斯特 + 因果线性啁啾相位 + 滤波相位平滑/浊音低频噪声抑制正则，以及 `per-channel` + 声码器 Post-GELU `pw2` 离群保护 INT8 导出
  - 在 `BZNSYP` 全量上跑完 `a10` `full_quality` 闭环训练并重新导出 ONNX/INT8 验收
- [ ] **Phase 2：真·流式前端闭环与双语精品音色**
  - 升级为**增量流式双语前端**（2–4 汉字共享滑窗，处理数字 TN / 多音字 / 三声变调 / 英文整词），向 `ChunkLookaheadBackbone.forward_committed_with_lookahead` 输出 `(committed_ids, tentative_lookahead_ids)`
  - 使用 15–25h 单音色精品双语语料 + Apache 2.0 教师离线补齐中英混读短板
  - 完成「多句连续流式输入不重置状态」跨句韵律与零接缝爆音验收
- [ ] **Phase 3：量化验证与三段式流式图导出**
  - 已完成离线两段式与三段式流式 ONNX（`text_step` → `frame_step` → `vocoder_step`）导出与 `per-channel` 混合精度 INT8 保护；后续在 `a10` 实机权重上完成 FP32 ↔ INT8 盲听无损验收
- [ ] **Phase 4：嵌入式 Linux C++17 纯 CPU 流式推理引擎**
  - C++17 流式前端 + ONNX Runtime / MNN 纯 CPU 调度 + ALSA/PCM 低延迟环形缓冲，在双核 ARM64 开发板实测达成 `RTF < 0.10`、`TTFA < 150 ms`
