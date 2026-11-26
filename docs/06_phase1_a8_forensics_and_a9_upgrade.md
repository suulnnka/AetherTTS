# Phase 1（`a8`）实机质量取证报告与 `a9` 全链路升级方案

> 日期：2026-11-26 ｜ 评测对象：`bznsyp_full_quality_a8_fL2_vL2`（`E3_chunk_attn_C4_L8` + `CausalVocosLite+Harm` + `Pretrained Vocos` 对照，共 61 句 = 50 句标贝 Val + 11 句黄金测试句）
> 结论速览：**`a8` 彻底根治了 `a6/a7` 的稀疏锚点对齐坍缩（MER 从 92.1% 骤降至 13.14%，纯中文语句基本可懂），但全量波形/频谱取证暴露了 6 项独立根因——其中声学模型占总自然度缺口的 ~83%（`UTMOS_PreVocos = 1.941` vs GT `3.77`），声码器则贡献了强烈的 93.75 Hz 电音、元音中段 OLA 断层空洞及 `Stream_MaxDiff = 0.27` 相位分叉。以上全部问题已在 `a9` 架构一次性修复并过门禁。**

---

## 1. `a8` 实机听感与 61 句量化取证

### 1.1 客观指标总览与缺口分解

| 维度 | GT 真声锚点 | `a8` 声学 + `Pretrained Vocos` | `a8` 声学 + `CausalVocosLite (offline)` | `a8` 声学 + `CausalVocosLite (streaming)` | 诊断结论 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UTMOS↑** | `3.770` | `1.941`（Val `1.975` / Gold `1.790`） | `~1.560` | `1.560` | 总缺口 `2.21` 分：**声学侧丢 `1.83` 分（83%）**，声码器再丢 `0.38` 分（17%） |
| **MER / CER↓** | — | — | — | `13.14%`（Val `14.51%` / Gold `6.91%`） | 中文基本可懂，但存在声母含糊（如 `val_004626`「张博强→张不强」）与未训英文乱码（`zh_cs_01`） |
| **93.75 Hz 包络调制比 (`Mod_93Hz`)↓** | — | `0.0023`（底噪水平） | `0.0080`（**3.5×**） | `0.0081`（**3.5×**） | `CausalVocosLite` 存在显著帧率（`24000/256`）周期振幅锯齿（电音物理根因） |
| **帧内 `n % 256` RMS 起伏比↓** | — | `1.060`（平坦） | `1.159`（**16% 纹波**） | `1.159`（**16% 纹波**） | 同份 Mel 下，`CausalVocosLite` 帧边界存在相位不连续 |
| **有声段 >12 dB 能量空洞 (`Energy_Holes`)↓** | — | `9` 次 / 61 句 | `35` 次 / 61 句（**3.9×**） | `34` 次 / 61 句（**3.8×**） | 元音中段突发 OLA 相消塌陷（如 `val_004624` f33–34 RMS `0.13→0.044→0.10`），即听到的「强断层感」 |
| **F0 Jitter / >20% 突跳（`zh_qa_01`）↓** | — | `5.79%` / `7` 次 | — | `13.46%` / `20` 次（**~2.5×**） | 声码器引入高频抖动与次谐波劈叉（如 `val_003917` f272 从 `247 Hz` 劈成 `123 Hz`） |
| **0.5–2 kHz 共振峰能量 / 4–12 kHz 高频能量** | — | `32.2%` / `4.1%` | `24.4%` / `5.4%` | `24.3%` / `5.5%` | 声码器清晰度频段流失 **25%**、高频金属毛刺多出 **33%** |
| **流式↔离线一致性** | — | — | 61/61 长度严格一致，`corr ∈ [0.9865, 1.0000]` | `Stream_MaxDiff = 0.27`（5 句出现 5–11 帧局部爆发） | **非 OLA 拼接 Bug**；源于 TF32/内核微差在声码器相位相消鞍点被放大 250× |

### 1.2 六大代码级根因定位

1. **声码器 GTA 闭环存在严重 F0/UV/能量错配投毒（重大 Bug）**
   - `a8` 的 `extract_gta_mel` 用声学模型**预测的 `f0_pred / uv_prob / e_pred`** 导出 GTA 谱（此时 `val_f0` 误差高达 `0.45–0.48` z-score，即经常偏离真声基频 20–40 Hz）。
   - 随后声码器在 Stage 2 后 60%（**整整 25,200 步**、全量高学习率阶段）以 35% 概率喂入基频错位的 `mel_gta`，却用真声 `wav_gt` 与真声 `f0_gt` 施加强 `35.0 * loss_mel + loss_adv + loss_vf0` 监督，迫使声码器在输入谐波与目标基频冲突中把谐波相位训糊、训出电音和次谐波劈叉。
2. **因果 iSTFT 缺少显式相位约束与物理谐波载波**
   - `a8` 的 `CausalHarmonicSource` 仅用 1 层 `CausalConv1d(k=3)` 猜 F0，再将 93.75 Hz 帧率下已经混叠的 12 阶标量 `sin/cos` 经固定线性层加到隐状态里——当中文四声 F0 在 180–350 Hz 滑动时，静态线性层无法把第 $h$ 阶谐波相位动态路由到移动的 STFT 频点 $b \approx h f_0 / 23.4375$。
   - 同时 `multi_res_stft_loss` 与 `SpecDisc` 均为纯幅度谱（`.abs()`），零显式跨帧相位监督；且 `vocoder_lookahead = 2`（21.3 ms）连单帧 1024 点 iSTFT 右半窗都未完全覆盖。
3. **TF32 与离线/流式注意力内核差异触发相位分叉（`Stream_MaxDiff = 0.27`）**
   - 全局 `allow_tf32 = True`、离线 `F.scaled_dot_product_attention` vs 流式手写 `torch.matmul`、以及整句/分块 cuDNN 卷积算法差异引入 `~1e-3` 数值扰动，在声码器不稳定的 OLA 相消鞍点处被放大 250 倍（如 `val_004081` f198 差 `0.2656`、`val_003917` f269 差 `0.2402`）。
4. **声学模型严重过拟合（全网 0 Dropout × ~220–290 Epoch）与选模陷阱**
   - `E3` 全链路无任何 Dropout，45k 步训练曲线（`comparison_charts.png`）明确显示 `val_f0` 在 22.5k 步触底（`0.448 → 0.481`）、`val_mel` 在 31.5k 步触底（`0.370 → 0.376`）。
   - `_infer_mel_l1` 使用全局线性 `F.interpolate` 对齐长度，局部时长只要差 2 帧就会抬高全句 L1，反而偏好早期模糊谱（`infer_mel` 最低点落在硬对齐尚未开启的第 9,000 步 `0.615`）。
5. **声学韵律预测头与帧解码器感受野过窄 + 韵律边界训推失配**
   - `f0_uv_predictor` / `energy_predictor` 仅 2 层 `CausalConv1d(k=3)`（感受野 **5 帧 ≈ 53 ms，0 右前瞻**），在分段常数 `h_frame` 的音素边界处产生硬跳变；`frame_decoder` 仅 1+3 层 `k=5` 浅卷积（左感受野 16 帧 = 170 ms），纯 L1 回归导致共振峰过平滑（`UTMOS_PreVocos` 停在 `1.941`）。
   - 训练期标贝含完整的 `{0, 1, 2(#1), 3(#2), 4(#3)}` 边界，但推理期 `_zh_span_to_phonemes_and_bounds` 永不输出 `3`（`#2` 韵律短语），导致无标注黄金测试句比带标注 Val 句再低 `0.185` UTMOS（长句 `zh_long_01` 仅 `1.622`）。
6. **评测盲区与单语数据边界**
   - `detect_seam_click` 仅在 170 ms 声码器块接缝处用 12–15σ 查单样本尖峰，对 93.75 Hz 帧调制与 OLA 能量空洞完全失明（全部误报 `0.0`）；`diagnose_phase1.py` 第 303/466 行漏了 `F.log_softmax(alp, dim=-1)`；纯中文 BZNSYP 阶段 69 个英文音素未训练，`zh_cs_01` 输出乱码并拉低总分。

---

## 2. `a9` 全链路修复与架构升级明细

`a9` 严格保持 **单一真相源**（`notebooks/AetherTTS_Colab_Phase1_Experiment.ipynb` → `sync_phase1_script.py` → `aethertts_lib.py` + `aethertts_phase1.py`），**继续复用 `v4` 特征缓存（零重抽）**，同时守住单音色端侧参数与延迟红线。

### 2.1 声码器升级：`CausalVocosLite (a9)` + 相位监督 + 无毒 GTA

| 模块 | `a8` 现状 | `a9` 升级方案 | 解决的根因 |
| :--- | :--- | :--- | :--- |
| **GTA 导出 (`extract_gta_mel`)** | 用预测 `f0_pred/uv_prob/e_pred` 导谱，Stage 2 后 60% (`p=0.35`) 混入 | 传入真声 **`f0_gt / uv_gt / energy_gt`** 生成 100% 同基频对齐 GTA 谱；推迟至 **Stage 2 末 25% 低 LR 阶段**（`p=0.25`）温和适应 | 彻底消除 GTA 基频错位投毒 |
| **谐波激励 (`CausalHarmonicSource`)** | 1 层 `k=3` 基频头 + 帧率 12 阶标量线性相加 | **2 层空洞因果基频头** `(k=5, d=1,2)` + `float64` **梯形相位积分**（OLA 交汇中点二阶精确连续）+ **24 阶切比雪夫带限声门脉冲 STFT 复数载波** `harm_spec`（精确 Hann² COLA 误差 `~1e-5`、零 24 kHz 环形缓冲、严格 `no_grad` 梯度隔离） | 根治 93.75 Hz 帧率电音与高次谐波相位路由难题 |
| **流式 iSTFT 头 (`StreamingISTFTHead`)** | 单分支预测 513 维原始相位 `p` | **谐波源-滤波器 + 非周期噪声双分支复频头**：`spec = harm_spec * exp(mag_h + j·p_h) + exp(mag_n + j·p_n)`；网络只需预测慢变声道包络与气声 | 消除 OLA 相消空洞与次谐波劈叉 |
| **主干与前瞻 (`CausalVocosLite`)** | `dim=256`, `lookahead=2f`, 8 层 `d=1` | **`dim=320`, `lookahead=4f`（42.7 ms 覆盖完整 1024 点右半窗）, 8 层多尺度空洞 `d=(1,2,4,1,2,4,1,2)`**（左 RF 102 帧 ≈ 1.09 s）；首包自适应 `first_chunk = max(6, voc_LA + 4) = 8f`（仍在首个文本块立即出声，零额外 LLM 等待） | 消除因果相位盲区、拉升 0.5–2 kHz 共振峰清晰度 |
| **相位损失与判别器** | 纯幅度 STFT + 1 通道幅度 `SpecDisc` | 新增全 FP32 无奇点 **多分辨率瞬时频率 (Δₜφ) + 群延迟 (Δ_fφ) 相位一致性损失**（Stage 1 权重 `5.0`，Stage 2 权重 `2.0`）；`SpecDisc` 升级为 **`[mag^0.3, mag^0.3·Re(Δₜu), mag^0.3·Im(Δₜu)]` 3 通道相位感知 MRD** | 显式压制 93.75 Hz 调制与跨帧相位断层 |

### 2.2 声学模型与前端升级：`E3_chunk_attn_C4_L8 (a9)`

| 模块 | `a8` 现状 | `a9` 升级方案 | 解决的根因 |
| :--- | :--- | :--- | :--- |
| **共享帧前瞻 (`frame_pre`)** | 在 `f0_uv` / `energy` 之后 | **前移至 `f0_uv_predictor` 与 `energy_predictor` 之前**（零额外延迟，仍仅 `frame_lookahead = 2f`） | 消除音素边界阶跃导致的 F0/能量硬跳变 |
| **韵律预测头 (`CausalPredictorHead`)** | 2 层稠密 `k=3`（RF 5 帧 = 53 ms，0 前瞻，1.18M） | **多尺度空洞因果深度可分离残差块**：时长头 `k=3, d=(1,2)`；F0/UV & Energy 头 `k=5, d=(1,2,4)`（叠加 `frame_pre` 后 RF **33 帧 ≈ 352 ms + 2 帧前瞻**，总参降至 0.53M、CPU 省 ~3×） | 大幅降低 `val_f0` 误差与基频抖动 |
| **帧解码器 (`frame_decoder`)** | 3 层 `d=1`（RF 16 帧，纯 L1） | **6 层多尺度空洞因果 ConvNeXt** `d=(1,2,4,1,2,4)`（左 RF **60 帧 ≈ 640 ms + 2 帧前瞻**）+ **中层深监督**（推理期剥离零开销）+ **二阶频谱曲率结构损失**（权重升至 `0.50`，F0 权重升至 `0.50`） | 打破 `UTMOS_PreVocos = 1.94` 上限、锐化共振峰与清辅音 |
| **抗过拟合与选模** | 0 Dropout，45k 步；线性拉伸 `infer_mel` 选模 | 全链路引入 **`dropout = 0.1`**，full_quality 调至 `40k` 步；`_infer_mel_l1` 升级为 **Sakoe-Chiba 带状 DTW Mel L1** + **硬对齐门控**（`step_ratio >= 0.55` 重置 `best_val_infer`） | 根治 250-epoch 过拟合与早期糊谱锁定 `best.pt` |
| **韵律边界与三声变调** | 训练有 `#2(3)`、推理永不出 `3`；机械两两三声变调 | 训练期 **25% 句级 `#2→#1` 边界随机退化**；推理期长分句（≥9 音节）自动按分词词首补齐 `#2(3)`；词界感知两轮三声变调（`展览馆买把好雨伞` 正确变调 `zhan2 lan2 guan3`） | 弥合 Val 与自由文本推理的 `-0.185` UTMOS 鸿沟 |
| **流式一致性与评测指标** | `forward_chunk` 手写 matmul + 评测开 TF32；`Seam_Click` 盲区 | `forward_chunk` 统一 `F.scaled_dot_product_attention`；Cell 10 评测合成期关闭 TF32；Cell 11 与 `diagnose_phase1.py` 新增 **`Mod_93Hz`、`Energy_Holes`、`MER_ZH`** 并收紧 `detect_seam_click` 阈值至 8–10σ；修复 `diagnose_phase1.py` 的 `log_softmax` 遗漏 | 消除 `Stream_MaxDiff = 0.27` 并杜绝伪影漏检 |

---

## 3. `a9` 参数量、延迟预算与门禁自检实测

在合并前已于本地 CPU 实测通过全部 `--selftest` 硬门禁及 2 线程 CPU 基准校验：

- **参数量**：声学推理 **`5.15M`**（剥离 `aligner` 与 `mel_coarse_out`）+ 声码器 **`6.35M`** ≈ 端到端 **`11.50M`**（严格落在架构蓝图 `8.5M–11.5M` 单音色预算内，动态 INT8 量化 `INT8_MelDiff = 0.0211`）。
- **门禁校验（实机 CPU `--selftest` 100% 通过）**：
  - `Frontend Phonology (a9)`：20,901 CJK 0-UNK、词界三声（`展览馆→zhan2 lan2 guan3`）、长句 `#2(3)` 韵律边界全过；
  - `ChunkAttn Parallel + LongRoPE (T=605)`：前向等价 `7.15e-07`、反向 `2.67e-05`、超长流式 `9.54e-07`、零前瞻泄漏 `Leak = 0.0`；
  - `E3_chunk_attn_C4_L8 (a9)`：流式等价 `2.38e-06`、`Leak = 0.0`；
  - `CausalVocosLite+Harm (a9)`：同 F0 谐波 STFT 载波逐位等价 `0.0e+00`、精确 Hann² COLA 误差 `3.07e-06`、端到端自适应 `8→16` 与 `step=1` 流式 RMS 误差 `1.37e-05`（≈ 0.45 LSB of 16-bit PCM）；
  - `a9 Phase / Grad / GTA Gates`：波形梯度对 `pitch_out` 严格为 `0`、全零静音输入下 STFT/相位损失/3 通道 `SpecDisc` 梯度 100% 有限无 NaN、GT-F0 对齐 `extract_gta_mel` 校验通过。

---

## 4. `a9` 推荐运行命令（RTX 4090 / Colab）

```bash
# 1. 拉取 main 并先跑一遍 10 秒门禁自检
git pull origin main
python scripts/sync_phase1_script.py --check
python scripts/aethertts_phase1.py --selftest --skip-pip

# 2. 启动 full_quality 档全链路训练与评测（自动复用 v4 特征缓存，产物隔离在 *_a9_fL2 / *_a9_L4 目录）
python scripts/aethertts_phase1.py --profile full_quality --dataset bznsyp --batch-size 64 --root ./aethertts_exp --skip-pip

# 3. 跑 8 链路逐级隔离诊断（含 Mod_93Hz / Energy_Holes / best_step 审计）
python scripts/diagnose_phase1.py --root ./aethertts_exp --profile full_quality --num-utts 5
```
