# AetherTTS 训练数据清单与使用策略

结构化机器可读清单见 [`datasets.yaml`](datasets.yaml)。本说明聚焦于**版权合规分级**、**中英混读数据瓶颈的破解思路**，以及各阶段的数据配比建议。

---

## 1. 数据集分级总览

### A. 商用友好核心语料（正式训练优先）

| ID | 数据集 | 语种 | 时长 | 说话人 | 采样率 | 协议 | 推荐用途 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `aishell3` | [AISHELL-3 (SLR93)](https://openslr.org/93/) / [HF](https://huggingface.co/datasets/SMIIP-lab/AISHELL-3) | 中文 | 85h | 218 | 44.1 kHz | **Apache 2.0** | 中文高保真多说话人底料；可精选 1–2 名标准音色作主音色种子 |
| `hifi_tts` | [Hi-Fi TTS (SLR109)](https://www.openslr.org/109/) | 英文 | 291.6h | 10 | 44.1 kHz | **CC BY 4.0** | 英文录音棚级主力底料（单人 17–40h+），优先用 `clean` 子集 |
| `libritts_r` | [LibriTTS-R (SLR141)](https://www.openslr.org/141/) / [ModelScope](https://www.modelscope.cn/datasets/pengzhendong/libritts_r) | 英文 | 585h | 2,456 | 24 kHz | **CC BY 4.0** | 原生 24 kHz 语音修复多说话人英文底料，扩充发音与韵律覆盖 |
| `ljspeech` | [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) / [Direct tar.bz2](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) | 英文 | 24h | 1 | 22.05 kHz | **Public Domain** | Phase 1 快速单人原型验证、消融实验与对齐排错 |
| `vctk` | [CSTR VCTK v0.92](https://datashare.ed.ac.uk/handle/10283/3443) | 英文 | 44h | 109 | 48 kHz | **CC BY 4.0** | 48 kHz 录音棚英文，补充声码器高频与口音泛化 |
| `emilia_yodas` | [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 多语（EN/ZH 等） | 114k h | 大量 | 24 kHz | **CC BY 4.0** | **仅限 `Emilia-YODAS` 子集**（严禁混入同仓库 CC BY-NC 的原始 Emilia）；需二次高阈过滤 |
| `ascend` | [CAiRE/ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND) | 中英混读 | 10.6h | 23 | 16 kHz | **CC BY-SA 4.0** | 注意 **SA 条款**；主要用作中英混读文本/停顿参考与评测，不直接作主音色声学底料 |

### B. 学术精品与研究对照语料（仅限调研/基线实验，商用必须隔离）

| ID | 数据集 | 语种 | 时长 | 说话人 | 采样率 | 协议 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `databaker_csmsc` | [标贝 CSMSC (BZNSYP)](https://www.data-baker.com/data/index/TNtts/) / [HF 镜像](https://huggingface.co/openspeech/BZNSYP) | 中文 | 12h | 1 | 48 kHz | 仅限非商用（商用需另购） | **调研阶段中文单人黄金基线**，自带 `#1/#2/#3/#4` 人工韵律边界与音素标注 |
| `emilia_original` | [Emilia 原始子集](https://huggingface.co/datasets/amphion/Emilia-Dataset) / [ModelScope](https://www.modelscope.ai/datasets/modelscope/Emilia-Dataset) | 中/英/多语 | 101k h | 大量 | 24 kHz | CC BY-NC 4.0 | 大规模自然口语与混读对照预训练 |
| `wenetspeech4tts` | [WenetSpeech4TTS](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS) | 中文 | 12.8k h | 大量 | 16 kHz | CC BY-NC 4.0 | 建议仅参考 `Premium`（~945h）子集 |
| `cs_dialogue` | [BAAI/CS-Dialogue](https://huggingface.co/datasets/BAAI/CS-Dialogue) | 中英混读 | 104h | 多人对话 | 16 kHz | CC BY-NC-SA 4.0 | 真实中英混读多轮对话分布分析与测试 |
| `magicdata_zh` | [MAGICDATA (SLR68)](https://www.openslr.org/68/) | 中文 | 755h | 1,080 | 16 kHz | CC BY-NC-ND 4.0 | 智能家居/问答指令域文本分布极佳 |
| `esd` | [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) | 中/英 | 29h | 20 | 16 kHz | 仅限科研申请 | 后续拓展情感可控时的参考 |

### C. 前端词典、多音字与评测基准

| 类型 | 名称与地址 | 协议 | 用途 |
| :--- | :--- | :--- | :--- |
| 中文多音字 | [CPP / g2pM](https://github.com/kakaobrain/g2pm) · [g2pW](https://github.com/GitYCC/g2pW) | Apache 2.0 | 训练轻量（~1–2M）流式多音字消歧模型 |
| 文本正则 TN | [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) | Apache 2.0 | 中英文数字、日期、金额、单位、符号正则化规则基准 |
| 英文发音词典 | [CMUdict](https://github.com/cmusphinx/cmudict) | BSD-2-Clause | 英文单词到 ARPAbet + 重音映射 |
| 客观评测集 | [Seed-TTS-Eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) | Benchmark | 评估中英 CER/WER、困难多音字及 UTMOS |

---

## 2. 单音色严控体量策略（拒绝盲目堆数据，防止训练时间失控）

本项目**只需要 1 个音色**。对 8M–12M 端侧小模型而言，直接把上百小时多说话人语料全量塞进去不仅会让训练时间膨胀数倍，还会严重**稀释小模型的容量**（被迫学几十上百个不用的声纹）。

经验标尺：
- `标贝 BZNSYP` 仅 **12 小时（10,000 句）**即可训出广播级单人中文；
- `LJSpeech` 仅 **24 小时（13,100 句）**即可训出近真人单人英文；
- `Paradee-8M` 仅用 **23.9 小时（12,000 句）**单音色蒸馏数据，在单卡/笔记本级算力上就达到了 UTMOS 4.41。

因此，AetherTTS **将训练集总量死锁在 15–25 小时以内**，按阶段精准取用：

| 阶段 | 数据源与严格体量上限 | 预期单卡训练耗时（参考 RTX 4090） | 目的 |
| :--- | :--- | :--- | :--- |
| **Phase 1：主干对比实验**（SWA / Mamba / RWKV / 基线） | **仅用 8–12 小时单人单语数据**<br>• 英文：`LJSpeech` 精选 10–12h（商用安全）<br>• 中文：`标贝 BZNSYP` 12h（仅限调研对比）或教师先试造 10h 中文 | 每组实验 **3–8 小时**即可收敛对比 | 快速低成本完成流式主干与右前瞻消融，避免算力浪费 |
| **Phase 2：正式单音色双语模型** | **总计严格控制在 20–25 小时（单音色黄金集）**<br>直接用 Apache 2.0 教师（Qwen3-TTS / Fun-CosyVoice 3）+ 1 个合规参考声纹（自录干净干声或 `AISHELL-3` 精选 1 名单人种子）离线合成并严格四重清洗：<br>• **中文 12–14h**（~12,000 句，覆盖问答、多音字、数字、三声变调）<br>• **英文 5–6h**（~4,500 句，覆盖高频音素与日常指令）<br>• **中英混读 3–5h**（~3,500 句，覆盖高频中英夹杂句式） | 全流程训练 **8–24 小时**内完成 | **零多说话人冗余**，模型 100% 容量服务于这 1 个音色，且天然解决开源界“无单人中英混读录音棚数据”的死结 |
| **可选：声码器高频预热**（仅当单音色声码器泛化不足时启用） | 严禁全量使用大库！最多从 `AISHELL-3` + `Hi-Fi TTS (clean)` 各**按同性别抽样 ≤ 10h**（合计 ≤ 20h）预热声码器，再切回 20h 单音色微调 | 额外约 6–10 小时 | 补充极高频齿音与复杂音素泛化，同时严格封顶算力 |

> **关键防坑**：`LibriTTS-R (585h)`、`Hi-Fi TTS (291h)`、`Emilia-YODAS (114k h)` 在本项目中**仅作为可选抽样池或参考源存档，严禁全量拉取训练**！

---

## 3. 统一数据预处理规范（进入实验阶段时执行）

所有进入训练管线的音频统一标准化为：
- **格式**：单声道（Mono）、**24,000 Hz**、16-bit PCM WAV（或无损 FLAC）
- **响度**：统一归一化至 `-20 LUFS`（峰值限制 `-1 dBFS`，防止削波）
- **首尾静音**：VAD 裁剪首尾过长静音，统一保留 `50–100 ms` 自然起止缓冲
- **时长过滤**：剔除 `< 0.8 s` 或 `> 12.0 s` 的异常切片（短切片占比 20% 以上，确保流式首包稳定与 batch 效率）
- **对齐标注**：采用 MFA（Montreal Forced Aligner）提取精确音素时长，配合 `RMVPE` / `Harvest` 提取 24 kHz 帧级连续基频（F0）与能量（Energy）
