# 统计检验方法产生高质量子集（D_ref）：完整流程文档

> 本文档描述项目中如何通过声学特征提取 + 统计假设检验 + 逐样本打分，从原始 CREMA-D 数据中筛选出高质量参考子集 `D_ref`，供后续 Phase 1 训练 Reference Model 使用。

---

## 1. 整体流程概览

```
原始音频 + metadata.csv
        │
        ▼
 ┌─────────────────────┐
 │ 1. 声学特征提取       │  feature_extraction.py → extract_features()
 │    (22 维特征)        │
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 2. 组级统计检验       │  stats_validation.py → run_all_validations()
 │    (gender/age/      │  三个任务 × 各自特征子集
 │     emotion)         │
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 3. 逐样本打分        │  scoring.py → assign_significance_flags()
 │    (sig_score 0~3)   │
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 4. 筛选高质量子集     │  scoring.py → rank_high_score_samples()
 │    sig_score >= 2    │  → high_score_audios.csv = D_ref
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 5. Phase 0 数据准备   │  phase0/pipeline.py → run_phase0_pipeline()
 │    将 D_ref 标记注入   │  --ref-manifest high_score_audios.csv
 │    生成 tokenized     │
 │    manifests          │
 └─────────────────────┘
```

---

## 2. 输入数据

- **数据集**：CREMA-D（Crowd-sourced Emotional Multimodal Actors Dataset）
- **元数据文件**：`crema_data_1000/metadata.csv`（1000 条样本）或完整集
- **必须列**：`audio_path`, `text`, `gender_label`, `age_label`, `emotion_label`
- **标签标准化**：
  - `gender_label` → `gender_label_norm`：小写化（`male` / `female`）
  - `age_label` → `age_label_bin`：通过映射表二分为 `young` / `old`
    - young = child, youth, teen, young
    - old = adult, senior, elder, old
  - `emotion_label` → `emotion_label_norm`：小写化（neutral, happy, sad, angry, fear, surprised）

**相关函数**：`utils.py` → `load_metadata()`, `normalize_label()`, `DEFAULT_AGE_MAP`

---

## 3. 声学特征提取（22 维）

**入口函数**：`feature_extraction.py` → `extract_features(metadata_df)` → 逐条调用 `extract_single_audio_features()`

### 3.1 完整特征列表

| 编号 | 特征名 | 含义 | 提取方式 |
|------|--------|------|----------|
| 1 | `duration_sec` | 音频时长（秒） | `len(y) / sr` |
| 2 | `f0_mean` | 基频均值 | `librosa.pyin` → 有效帧均值 |
| 3 | `f0_std` | 基频标准差 | 同上 |
| 4 | `f0_min` | 基频最小值 | 同上 |
| 5 | `f0_max` | 基频最大值 | 同上 |
| 6 | `f0_range` | 基频范围 | `f0_max - f0_min` |
| 7 | `energy_mean` | RMS 能量均值 | `librosa.feature.rms` |
| 8 | `energy_std` | RMS 能量标准差 | 同上 |
| 9 | `energy_range` | RMS 能量范围 | `energy_max - energy_min` |
| 10 | `tempo_bpm` | 节奏速度 | `librosa.feature.tempo` |
| 11 | `pause_ratio` | 静音占比 | `librosa.effects.split` |
| 12 | `spec_centroid_mean` | 频谱质心均值 | `librosa.feature.spectral_centroid` |
| 13 | `spec_bandwidth_mean` | 频谱带宽均值 | `librosa.feature.spectral_bandwidth` |
| 14 | `spec_rolloff95_mean` | 频谱滚降点（95%）均值 | `librosa.feature.spectral_rolloff` |
| 15 | `spec_flux_mean` | 频谱通量均值（onset strength） | `librosa.onset.onset_strength` |
| 16 | `spec_tilt_proxy` | 频谱倾斜代理值 | Mel 高低频段能量差 |
| 17 | `mfcc_mean` | MFCC 均值 | `librosa.feature.mfcc(n_mfcc=13)` |
| 18 | `mfcc_std` | MFCC 标准差 | 同上 |
| 19 | `mfcc_delta_std` | MFCC 一阶差分标准差 | `librosa.feature.delta` |
| 20 | `hnr_mean` | 谐噪比均值 | `parselmouth`（可选） |
| 21 | `jitter_local` | 基频微扰 | `parselmouth`（可选） |
| 22 | `shimmer_local` | 振幅微扰 | `parselmouth`（可选） |

### 3.2 各任务实际使用的特征子集

虽然提取了 22 维特征，但**统计检验只使用其中 8 个**：

| 任务 | 使用特征 |
|------|----------|
| **Gender**（性别） | `f0_mean`, `spec_centroid_mean` |
| **Age**（年龄） | `f0_mean`, `pause_ratio` |
| **Emotion**（情绪） | `f0_range`, `energy_range`, `spec_flux_mean`, `mfcc_delta_std` |

---

## 4. 组级统计检验

**入口函数**：`stats_validation.py` → `run_all_validations()`

### 4.1 检验函数：`run_two_group_test()`

对两组数值进行假设检验，自动选择检验方法：

1. **样本量检查**：任一组 < 5 条 → 标记为 `insufficient_samples`，不做检验
2. **正态性预检**（Shapiro-Wilk，样本量 ≤ 5000 时）：
   - 两组都通过（p ≥ 0.05）→ 使用 **Welch t-test**（不等方差 t 检验）
   - 否则 → 使用 **Mann-Whitney U**（非参数秩检验，双侧）
3. **显著性判定**：`p < alpha`（默认 `alpha = 0.05`）

> 实际运行中，CREMA-D 数据基本走的是 Mann-Whitney U。

### 4.2 三个验证任务

#### 4.2.1 性别验证：`validate_gender()`

- **分组**：`male` vs `female`
- **检验特征**：`f0_mean`, `spec_centroid_mean`
- **每个特征做一次 two-group test**
- **同时为每个性别组、每个特征构建区间**（用于后续逐样本判定）

#### 4.2.2 年龄验证：`validate_age()`

- **分组**：`young` vs `old`
- **检验特征**：`f0_mean`, `pause_ratio`
- **逻辑同上**

#### 4.2.3 情绪验证：`validate_emotion()`

- **分组策略**：one-vs-rest（每个情绪类别 vs 其余所有样本）
- **有效情绪**：neutral, happy, sad, angry, fear, surprised
- **检验特征**：`f0_range`, `energy_range`, `spec_flux_mean`, `mfcc_delta_std`
- **每个情绪 × 每个特征 = 一次 two-group test**
- 例如 5 个情绪 × 4 特征 = 20 项检验

### 4.3 区间构建：`_build_group_intervals()`

为每个组（如 male/female/young/old/angry/happy/...）在每个特征上构建"典型值区间"：

- **std 方法**（默认）：`[mean - k × std, mean + k × std]`
  - `k = std_multiplier`（默认 1.5，宽松版用过 2.0）
- **iqr 方法**（可选）：`[Q1 - k × IQR, Q3 + k × IQR]`

**产出**：`stats_cache` 字典，包含每个任务的检验结果 + 每个组的区间参数。

---

## 5. 逐样本打分

**入口函数**：`scoring.py` → `assign_significance_flags(feature_df, stats_cache)`

对每一条样本，根据其标签和特征值，判断是否"符合该标签的统计规律"。

### 5.1 `gender_sig`（0 或 1）

**函数**：`assign_gender_sig()`

条件（同时满足）：
1. 主特征 `f0_mean` 的组级检验显著（`p < 0.05`）
2. 该样本的 `f0_mean` 值落入其所属性别组的区间 `[lower, upper]`

→ 满足则 `gender_sig = 1`，否则 `0`。

### 5.2 `age_sig`（0 或 1）

**函数**：`assign_age_sig()`

- 对 `f0_mean` 和 `pause_ratio` 分别投票：
  - 该特征组级检验显著 **且** 样本值落入对应年龄组区间 → 记 1 票
- **票数 ≥ 1** → `age_sig = 1`

### 5.3 `emotion_sig`（0 或 1）

**函数**：`assign_emotion_sig()`

- 针对样本所属情绪标签的 one-vs-rest 检验结果
- 在 4 个特征（`f0_range`, `energy_range`, `spec_flux_mean`, `mfcc_delta_std`）中：
  - 该特征检验显著 **且** 样本值落入该情绪组区间 → 记 1 匹配
- **匹配数 ≥ 2** → `emotion_sig = 1`

### 5.4 总分计算

```
sig_score = gender_sig + age_sig + emotion_sig    （范围 0~3）
```

辅助排序字段：
- `emotion_match_count`：情绪任务中满足条件的特征数（0~4）
- `age_vote_count`：年龄任务中满足条件的特征数（0~2）

---

## 6. 高质量子集筛选

**函数**：`scoring.py` → `rank_high_score_samples(scored_df, high_score_threshold=2)`

1. 按 `[sig_score, emotion_match_count, age_vote_count, duration_sec]` 降序排列
2. 筛选 `sig_score >= 2` 的样本 → 输出为 `high_score_audios.csv`

**这就是最终的高质量子集 `D_ref`。**

### 6.1 实际结果（CREMA-D 1000 样本，std_multiplier=1.5）

| sig_score | 样本数 |
|-----------|--------|
| 0 | 21 |
| 1 | 130 |
| 2 | 74 |
| 3 | 775 |
| **≥ 2（D_ref）** | **849** |

---

## 7. 从 D_ref 到 Phase 0 的衔接

`high_score_audios.csv` 作为 `--ref-manifest` 参数传入 Phase 0：

```bash
python phase0_main.py \
    --train-manifest crema_data_1000/metadata.csv \
    --ref-manifest crema_outputs_1000_tight_std15/high_score_audios.csv \
    --outdir phase0_outputs
```

Phase 0 的 `_build_ref_membership()` 通过 `sample_id` 或 `audio_path` 匹配，将 `high_score_audios.csv` 中的样本标记为 `in_ref=True`、`split="ref"`，其余标记为 `split="train"`。

最终产出：
- `phase0_manifest_ref.jsonl`：高质量参考子集（已 tokenized 为 `(y, μ)` 格式）
- `phase0_manifest_train.jsonl`：全部训练数据

---

## 8. 关键函数速查表

| 文件 | 函数 | 作用 |
|------|------|------|
| `feature_extraction.py` | `extract_single_audio_features()` | 从单条音频提取 22 维特征 |
| `feature_extraction.py` | `extract_features()` | 批量提取所有样本特征 |
| `stats_validation.py` | `run_two_group_test()` | 自动选择 t-test 或 Mann-Whitney U |
| `stats_validation.py` | `_build_group_intervals()` | 构建各组各特征的区间 |
| `stats_validation.py` | `validate_gender()` | 性别组级检验 |
| `stats_validation.py` | `validate_age()` | 年龄组级检验 |
| `stats_validation.py` | `validate_emotion()` | 情绪组级检验（one-vs-rest） |
| `stats_validation.py` | `run_all_validations()` | 运行全部三个任务的组级检验 |
| `scoring.py` | `assign_gender_sig()` | 逐样本性别显著性判定 |
| `scoring.py` | `assign_age_sig()` | 逐样本年龄显著性判定 |
| `scoring.py` | `assign_emotion_sig()` | 逐样本情绪显著性判定 |
| `scoring.py` | `assign_significance_flags()` | 汇总三维打分 |
| `scoring.py` | `rank_high_score_samples()` | 排序并筛选高分子集 |
| `utils.py` | `load_metadata()` | 加载并标准化元数据 |
| `main.py` | `main()` | 端到端入口 |
| `phase0/pipeline.py` | `run_phase0_pipeline()` | Phase 0 数据准备，注入 D_ref |

---

## 9. 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha` | 0.05 | 统计检验显著性水平 |
| `--interval-method` | `std` | 区间构建方法（`std` 或 `iqr`） |
| `--std-multiplier` | 1.5 | std 方法的区间宽度系数 k |
| `--iqr-multiplier` | 1.5 | iqr 方法的区间宽度系数 k |
| `high_score_threshold` | 2 | 高质量子集的 sig_score 最低阈值 |

收紧 `std_multiplier`（如从 2.0 → 1.5）会减少高分样本数量，提高筛选严格度。

---

## 10. 设计思路总结

本方法的核心思想是：**一条"高质量"语音样本应该在性别、年龄、情绪三个维度上都表现出与其标签一致的声学特征**。

1. **组级检验**确认某个特征在某个任务上确实有区分力（避免使用无区分度的特征）
2. **区间判定**确认单条样本在该特征上是"典型的"（不是异常值/噪声）
3. **多维投票**综合三个维度的判定结果，给出 0~3 的综合分数
4. 分数 ≥ 2 的样本被选为高质量子集，用于训练 Reference Model

这种方法不依赖人工主观评分，完全基于可重现的统计检验，适合大规模数据集的自动化质量筛选。
