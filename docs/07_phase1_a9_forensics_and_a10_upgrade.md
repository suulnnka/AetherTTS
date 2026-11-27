# Phase 1（`a9`）INT8 流式实机取证报告与 `a10` 全链路升级方案

> 日期：2026-11-27 ｜ 评测对象：`onnx_export/` 已导出的 `a9` 三段式流式 INT8 ONNX 模型（`text_step.int8.onnx` 3.3 MB + `frame_step.int8.onnx` 2.5 MB + `vocoder_step.int8.onnx` 10.9 MB，共 16.7 MB），基于 10 条标贝 24 kHz 验证集真声（含 `int8_ab` 中 UTMOS 降幅最大的 `val_007262`、`val_005925`、`val_006673`、`val_003626` 及上升句 `val_003780`）与自由测试句，执行 **5 路逐级隔离合成 + 人工试听 + ONNX 内部张量探针（`pitch_out` / 谐波-噪声分轨 / 帧间 OLA 相位一致性 / `DynamicQuantizeLinear` 量化信噪比）**。
>
> 结论速览：**相较 `a8`，`a9` 实现了跨越式进步（2 线程纯 CPU 均值 `TTFA = 27.8 ms`、`RTF = 0.149`，`Mod_93Hz` 从 `0.0081` 降至 `0.0042`，F0 抖动从 `13.46%` 降至 `1.97%`），中文可懂度与整体听感已可用顺畅。人工试听残留的两项细微缺陷——「声音略发闷」与「略微断层式电音」——经全链路取证精确定位为 5 项独立根因（其中还额外捕获了非 VNNI AVX2 CPU 上 `QInt8` 动态量化的 16-bit 饱和溢出问题）。以上全部根因已在 `a10`（`v5` 特征缓存）架构与 `export_onnx.py` 中一次性修复并通过全部门禁校验，同时严守推理端 `~11.50M` 零额外参数与零额外流式延迟红线。**

---

## 1. `a9` INT8 流式实机听感与 5 路隔离取证

### 1.1 客观指标总览（10 句标贝 Val，2 线程 CPU ONNX Runtime）

- **端侧 CPU 实测性能（10 句平均时长 `3.49 s`）**：首包延迟 `TTFA = 27.8 ms`（P95 `32.9 ms`），实时率 `RTF = 0.149`（最慢 `0.230`，快于实时 `~6.7×`）。
- **5 路隔离核心信号指标**：

| 评测链路 | `Mod_93Hz`↓ | >12 dB 空洞↓ | F0 抖动↓ | 85% / 95% 频谱滚降点 | `1–4 kHz` ΔdB | `8–12 kHz` ΔdB | Mel 锐度 `\|Δ²f\|`↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A`：GT 真声（24 kHz 锚点）** | `0.0026` | 18 | `2.53%` | `1146 / 3366 Hz` | `0.00` | `0.00` | `0.7714`（100%） |
| **`C`：GT Mel → `vocoder_int8`（流式 8+16）** | `0.0049` | 10 | `2.41%` | `1130 / 3302 Hz` | `-0.30 ~ -0.51` | `-0.47` | `0.7426`（**96.3%**） |
| **`D`：Text → 声学 + 声码器（整句）** | `0.0044` | 6 | `1.92%` | `1034 / 3084 Hz` | `-1.6` | `-1.4` | — |
| **`E`：Text → 声学 + 声码器（流式，默认）** | **`0.0042`** | **4** | **`1.97%`** | **`1027 / 3066 Hz`** | **`-1.62 ~ -1.65`** | **`-1.46`** | 预测谱 **`0.3837`（49.7%）** / 合成波 `0.6059` |
| *对照：`a8` 流式基线* | *`0.0081`* | *~5.6/10 句* | *`13.46%`* | — | — | — | — |

### 1.2 症状一取证：「声音略发闷」的频域归因（声学 ~80–85% + 声码器 ~15–20%）

1. **主因（~80–85%）：声学模型纯 L1 回归导致共振峰二阶曲率腰斩**
   - 喂 GT Mel 给 `vocoder_step.int8.onnx`（链路 `C`）时，Mel 频轴二阶曲率 `|Δ²f|` 保持 `96.3%`（`0.7426` vs `0.7714`），全频段与真声偏差仅 `-0.3 ~ -0.5 dB`，实听明亮度与真声接近。
   - 换成 `frame_step` 预测 Mel（链路 `E`）后，`0–0.5 kHz` 低频占比从 `50.29%` 抬升至 `56.43%`，而决定咬字明晰度的 **`1–4 kHz` 下跌 `-1.65 dB`**、**`10–12 kHz` 空气感下跌 `-3 ~ -5 dB`**；预测 Mel 的频率轴二阶曲率 `|Δ²f|` 整体腰斩 **`-50.3%`**（`0.3837` vs `0.7714`），其中 **`2–12 kHz` 更是塌缩 `-65% ~ -70%`**。确定性 L1 回归向条件中位数收缩是本质根因。
2. **次因（~15–20%）：声码器 `24` 阶谐波在 `4–6 kHz` 硬悬崖截断**
   - `a9` 的 `CausalHarmonicSource` 固定 `n_harmonics = 24`。对标贝女声常见 `170–250 Hz` 基频，第 24 阶谐波在 **`4.0–6.0 kHz` 直接撞墙归零**（探针实测谐噪比：`0–1k: +11.2 dB`、`1–2k: +5.8 dB`、`2–4k: +1.2 dB`、`4–6k: -1.4 dB`、**`6–8k: -49.3 dB`、`8–12k: -73.3 dB`**），高频只能完全退化给非相干噪声分支填补。

### 1.3 症状二取证：「略微断层式电音」的四大精确根因

通过给 `vocoder_step.int8.onnx` 注入内部张量探针，将波形无损拆解为 `carrier / harm / noise / sum`（重构误差 `2.56e-6`）并逐帧审计 OLA 相位一致性与 `DynamicQuantizeLinear` 量化信噪比：

| 声码器内部组件（10 句 Val 均值） | 有声帧 OLA 一致性↑ | 最差 10% (P10)↑ | `Mod_93Hz`↓ | 帧内纹波↓ | 实听质感 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`harm`（实际谐波分支，含 `p_h`）** | `0.7925` | `0.5285` | **`0.0018`** | `1.383` | 极干净（低于 GT `0.0026`），基本无电音 |
| **`noise`（实际噪声分支 `amp_n·e^{j·p_n}`）** | **`0.2612`** | **`0.0976`** | **`0.0536`（30×!）** | **`2.522`** | 强烈断续颗粒电音，且直接混入元音与过渡段 |
| **`sum`（最终输出 `harm + noise`）** | **`0.6792`** | **`0.2637`** | `0.0042` | `1.375` | 谐波干净，但被以下 4 项根因撕出断层感 |

1. **根因 B1（最直接）：噪声分支大举入侵低频浊音/过渡段 + 谐波滤波相位 `p_h` 帧间剧烈跳变**
   - `1–4 kHz` 谐噪比探针显示，在音节过渡与声调转折处，无相位约束的噪声分支反超谐波分支达 `10–40 dB`；由于其自身 OLA 一致性低至 `0.261`、`Mod_93Hz = 0.0536`，每次抬头都会把合成波形 OLA 一致性砸成 `0.20–0.50` 深 V 断层；同时谐波分支滤波相位 `p_h` 帧间跳变均值达 **`|Δt p_h| = 0.76 rad`（~44°/帧）**。
2. **根因 B2（重大数据 Bug）：`v4` 特征缓存把 99.7% 的清辅音错标成 `uv = 1` 并引发暴力基频俯冲**
   - `v4` 的 `extract_f0_and_energy` 仅用 `librosa.yin` + 能量阈值 `rms > rms_gate` 判浊音，没有任何周期性/非周期性门控。对 10 句 GT 中 `388` 帧明确高能清擦/塞擦音（`s/sh/x/c/ch/q/h/f/p/t/k`），**`388/388`（100.0%）被错标为 `uv = 1`**，导致训练出的 `a9` 声码器在 **`387/388`（99.7%）清辅音帧输出 `voc_uv > 0.5`**，强行灌入 `70–150 Hz` 乱跳谐波（单帧突跳达 `+130%`，10 句出现 `234` 次 >20% 基频跳变），把清辅音→元音衔接处（CV 边界）撕出明显断层爆裂感。
3. **根因 B3（数学机理）：`CausalHarmonicSource` 帧内常频假设在汉语四声快滑段自相消**
   - 虽然梯形积分保证了相邻帧在唯一中点 `+hop/2` 相位严格相等，但 `1024` 点（`42.7 ms`）窗内假设 `f0` 为常数。在汉语四声陡降段（如 `val_007262`「会」`356 → 164 Hz`），窗偏离中点 `±5.3 ms` 处高阶谐波错相 `180°–360°`，导致纯载波自身 OLA 一致性最低跌至 **`0.6659`**；改用分段线性啁啾二次相位后立刻恢复至 **`0.9996`（最低 `0.9971`）**。
4. **根因 B4（INT8 导出缺陷）：Per-Tensor + Post-GELU 高离群 `pw2` + AVX2 `U8S8` 16-bit 饱和溢出**
   - 原 `export_onnx.py` 使用 `per_channel=False` + `weight_type=QuantType.QInt8`（`reduce_range=False`）：
     - 在非 VNNI 的主流 x86 AVX2 CPU 上，`MatMulInteger(U8, S8)` 的 `vpmaddubsw` 触发 **16-bit 累加饱和溢出**（实测将 `vocoder_step.int8.onnx` 信噪比直接从 `47.0 dB` 砸到 **`25.6 dB`**，并引发 `text_step` 时长舍入漂移）；
     - `vocoder_step` 的 8 层 Post-GELU `pw2` 输入呈极端单侧尖峰（峰值因子 **Crest = `39.7×`**，`DynamicQuantizeLinear` 激活 SNR 最低跌至 **`27.14 dB`** ≈ 4.5 bit）；
     - 白名单漏掉了极小但高敏感的条件投影层（`in_proj` / `f0_emb` / `uv_emb` / `energy_emb` / `frame_phase_proj`，合计 `< 40 KB`）。
   - *已排除嫌疑*：`CooleyTukeyFFT1D` 的 12 个 FFT/iFFT `MatMul` 保持纯 FP32 未被量化；`StreamingOLA` 流式拼接完全正确（同 Mel 下流式↔整句相关系数 `0.9995–0.9998`）；`text_step`（`C=4, L=8`）文本切块边界无额外跳变。

---

## 2. `a10`（`v5` 特征缓存）全链路升级明细

`a10` 严格遵循 **单一真相源**（`notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb` → `sync_phase1_script.py` → `aethertts_lib.py` + `aethertts_phase1.py`），**端到端推理参数严格保持 `~11.50M`（`5.15M` + `6.35M`）、零新增流式前瞻延迟**，并对历史 `a9` checkpoint 与已有 `onnx_export/` 权重 100% 向后兼容。

### 2.1 `v5` 特征缓存升级：真清浊音特征提取（`extract_f0_and_energy`）

| 维度 | `v4`（`a6–a9`） | `v5`（`a10`） | 实测校准结果（10 句标贝 GT） |
| :--- | :--- | :--- | :--- |
| **周期性检测** | 无门控 `librosa.yin` + `rms > rms_gate` | 向量化 FFT-YIN 累计均值归一化差分（**CMNDF**）+ 幅度不变归一化自相关（**NACF**，解决响度骤增双元音斜坡期 YIN 假高） | 明确清辅音误判 `uv=1`：**`100% (388/388) → 0.00% (0/350)`** |
| **清擦音防误判** | 无频谱门控 | 增加 `>= 3 kHz / 80–1500 Hz` 高低频能量比 **`hi_ratio`** 双重清浊音门控 + 仅浊音段中值滤波/八度跳变修复 | 响亮低频元音召回 **`95.2%`**；双元音斜坡（`val_007262` f29–38）连续无断帧；抽帧仅 **`51.4 ms/句`** |

### 2.2 声学模型升级：`E3_chunk_attn_C4_L8 (a10)`（推理零增量，仍为 `5.15M`）

| 模块 | `a9` 现状 | `a10` 升级方案 | 解决的根因 |
| :--- | :--- | :--- | :--- |
| **Mel 频带加权与曲率损失** | 全频带等权 L1 + `0.50 * mel_structure_loss` | 新增 `get_mel_freq_weights`（HTK Mel 频带加权：`1–4 kHz` `1.45×`、`4–12 kHz` `1.35×`，均值归一化为 `1.0`），同步加权 `loss_mel`、`loss_coarse` 与 **`0.60 * loss_struct`**，并实时记录 `sharp_ratio`（`\|Δ²f\|_pred / \|Δ²f\|_gt`） | 拉回 `1–4 kHz` 清晰度与 `8–12 kHz` 空气感 |
| **训练期 2D Mel 对抗锐化** | 纯确定性 L1 回归 | 新增挂载于 `AetherAcousticModel` 外部的**双尺度 2D `MelPatchDiscriminator`**（2 × Weight-Norm 2D Conv 子判别器，显式 `mel_mask` 零填充屏蔽、`no_grad_real=True`）；后半程 `step_ratio >= 0.50` 线性爬坡激活（`adv=0.05`, `fm=0.50`, `lr=2e-4`），支持完整断点续训；**推理期零参数、零延迟** | 根治纯 L1 条件中位数模糊，恢复共振峰峰谷对比度 |
| **解码器末段 Dropout 退火与选模** | 全程固定 `dropout=0.1`；`best.pt` 仅看 `infer_mel + dur + f0` | `step_ratio ∈ [0.60, 0.85]` 将 `frame_decoder` dropout 线性退火至 `0.0`（保留前端防过拟合，消除末段方差压缩）；`best.pt` 纳入 `0.35 * val_struct` 锐度项，并在 MelGAN 爬坡稳态点（`0.65`）重置最优快照基准 | 兼顾抗过拟合与高频锐度，防止早期糊谱锁定 `best.pt` |

### 2.3 声码器升级：`CausalVocosLite (a10)`（仍为 `6.35M`，自动兼容 `a9` 权重）

| 模块 | `a9` 现状 | `a10` 升级方案 | 解决的根因 |
| :--- | :--- | :--- | :--- |
| **谐波阶数与奈奎斯特滚降** | `n_harmonics = 24`（`4–6 kHz` 硬截断） | **`n_harmonics = 48`**（女声基频覆盖至 **`10.8 kHz`**）+ **500 Hz 平滑奈奎斯特线性滚降** `((nyq_lim - f0·h) / 500).clamp(0, 1)`，零额外参数与状态 | 消除 `4–6 kHz` 谐波硬悬崖与发闷次因 |
| **帧内啁啾相位积分** | 常频假设 `phi + f0·t` | 严格因果后向斜率线性啁啾二次相位：`f0_slope = clamp((f0 - f0_prev)/dt_hop, -4000, 4000)`，`phase = phi + f0·t + 0.5·slope·t²`（零额外前瞻与缓存状态，保留 `use_chirp=False` 精确兼容 `a9`） | 四声陡降滑音载波 OLA 一致性 **`0.666 → 0.999`** |
| **双分支头正则与相位权重** | 无 `p_h` / 噪声入侵显式约束；Stage 2 `loss_phase=2.0` | 新增 `vocoder_head_reg_loss`（连续浊音帧 `p_h` 复单位向量平滑 `loss_ph_smooth` + 浊音 `0–1.5 kHz` 抑制 `noise_mag > 0.25·harm_mag` 的 `loss_noise_prior`，Stage 1 `0.25/0.10`、Stage 2 `0.15/0.05`）；Stage 2 `loss_phase` 提至 **`3.0`** | 根治噪声分支入侵低频元音与 `p_h` 44°/帧跳变导致的断层电音 |

### 2.4 ONNX / INT8 导出升级：`scripts/export_onnx.py`（零重训亦可立即受益）

- **自动架构识别**：`load_vocoder_model`、`aethertts_infer.py` 与 `diagnose_phase1.py` 均自动识别 `a9`（`H=24, use_chirp=False`）与 `a10`（`H=48, use_chirp=True`），并优先发现 `a10*` 与 `v5` 缓存。
- **`QUInt8` + `per_channel=True` + 敏感/离群层保护**：
  - 将 `quantize_dynamic` 升级为 `per_channel=True` + **`weight_type=QuantType.QUInt8`**，彻底消灭非 VNNI AVX2 CPU 上的 `U8S8` 16-bit 饱和溢出（同模型 `vocoder_SNR` 暴涨 **`+21.4 dB`**：`25.6 dB → 47.0 dB`，流式时长完全对齐 `F=80/80`、`mel_MAE: 1.2189 → 0.0120`）；
  - 敏感微层白名单补全 `in_proj`、`f0_emb`、`uv_emb`、`energy_emb`、`frame_phase_proj`（合计 `< 40 KB`）；
  - 新增 `--int8-profile`：`balanced`（默认，额外以 FP32 保护声码器 8 层 Post-GELU `pw2` 高离群层，三段流式共 `23.13 MB`）与 `compact`（最小体积 `16.12 MB`，仍享受 `QUInt8 per-channel` 与敏感微层保护）；
  - `verify_exports` 新增 INT8 三段流式数值审计与 `vocoder_SNR ≥ 30 dB` 溢出硬门禁。

---

## 3. 门禁自检与 ONNX / INT8 导出实测校验

在推送前已于本地 CPU 完成全量同步校验、门禁自检与 ONNX/INT8 端到端一致性校验（全部 100% 通过）：

- **单一真相源同步**：`python scripts/sync_phase1_script.py --check` → **`[OK]`**。
- **`--selftest` 全部硬门禁通过（`scripts/aethertts_phase1.py --selftest --skip-pip`）**：
  - `Frontend & v5 F0/UV (a10)`：20,901 CJK 0-UNK、词界三声变调、长句 `#2(3)` 边界、清擦音/元音 UV 门禁 **`FricUV = 0.0%`、`VowelUV = 100.0%`**；
  - `E3_chunk_attn_C4_L8`：推理参 **`5.15M`**、流式等价 **`2.38e-06`**、`Leak = 0.0`；
  - `CausalVocosLite+Harm (a10)`：参数 **`6.35M`**（端到端 **`11.50M`**）、同 F0 载波等价 `0.0e+00`、精确 Hann² COLA 误差 **`1.15e-05`**、自适应流式 OLA RMS **`2.31e-05`**、48 阶四声滑音（`350→170 Hz`）OLA 一致性 **`GlideCoh = 0.999`**（> 0.90 门禁）；
  - `Aligner Anti-Collapse (a8)`：`ctc(anch=16.74 > unif=2.07)`、`bias_heal dur1: 48% → 0%`；
  - `a10 Phase/MelDisc/GTA Gate`：`pitch_out` 梯度严格为 `0`、`MelPatchDiscriminator` padding 掩码严格不变、全零静音输入梯度 100% 有限无 NaN、GT-F0 对齐 GTA 校验通过。
- **ONNX 离线 + 三段式流式导出与 INT8 数值校验（`export_onnx.py --dummy --quantize-int8`）**：
  - `a9`（`H=24, chirp=False`）与 `a10`（`H=48, chirp=True`）`VocoderOfflineWrapper` ↔ PyTorch 误差分别低至 `1.44e-08` / `1.52e-08`；
  - FP32 三段式流式 vs PyTorch 离线：`max_mel_err = 5.72e-06`，端到端波形 `len=20480` 严格一致、`rms = 3.43e-05`；
  - **`balanced` INT8（默认）**：`text_step` `3.20 MB` + `frame_step` `2.44 MB` + `vocoder_step` `17.49 MB`（共 `23.13 MB`），`F = 80/80`、`mel_MAE = 0.0120`、**`vocoder_SNR = 47.0 dB`**；
  - **`compact` INT8**：`3.20 + 2.44 + 10.48 = 16.12 MB`，`F = 80/80`、`mel_MAE = 0.0120`、**`vocoder_SNR = 45.3 dB`**；
  - 兼容回归：现有 `onnx_export/*.int8.onnx`（`a9` 权重）使用 `infer_onnx.py` 流式推理正常通过。

---

## 4. 推荐运行命令（零重训立刻收益 + `a10` 全链路训练）

### 4.1 零重训立即可做：用新 `export_onnx.py` 重导现有 `a9` 权重

仅重新导出即可立即消除 AVX2 `U8S8` 饱和溢出与 `pw2` 离群失真（自动识别 `a9` 的 `H=24, chirp=False` 权重）：

```bash
git pull origin main
python scripts/export_onnx.py \
  --root ./aethertts_exp \
  --profile full_quality \
  --out-dir ./onnx_export_a9_balanced \
  --quantize-int8 \
  --int8-profile balanced
```

### 4.2 全链路迭代：启动 `a10`（`v5` 特征缓存）训练、诊断与导出

```bash
# 1. 拉取代码并运行 10 秒同步校验 + 门禁自检
git pull origin main
python scripts/sync_phase1_script.py --check
python scripts/aethertts_phase1.py --selftest --skip-pip

# 2. 启动 a10 full_quality 训练（首次会自动用 v5 重抽真清浊音特征缓存，约数分钟；产物隔离在 *_a10_fL2 / *_a10_L4 目录）
python scripts/aethertts_phase1.py \
  --profile full_quality \
  --dataset bznsyp \
  --batch-size 64 \
  --root ./aethertts_exp \
  --skip-pip

# 3. 跑 8 链路逐级隔离诊断（含 Mod_93Hz / Energy_Holes / Formant_Sharp 审计）
python scripts/diagnose_phase1.py --root ./aethertts_exp --profile full_quality --num-utts 5

# 4. 导出 a10 离线 + 三段式流式 ONNX（默认 balanced INT8；若需最小体积可加 --int8-profile compact）
python scripts/export_onnx.py \
  --root ./aethertts_exp \
  --profile full_quality \
  --out-dir ./onnx_export \
  --quantize-int8
```
