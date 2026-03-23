# CREMA-D 1000样本统计验证结果说明

## 1. 结果对应的数据与文件

本说明对应以下输出目录与文件：

- `crema_outputs_1000/summary.json`
- `crema_outputs_1000/group_test_results.csv`
- `crema_outputs_1000/audio_significance_results.csv`
- `crema_outputs_1000/high_score_audios.csv`
- `crema_outputs_1000/features.csv`

本次样本为从 CREMA-D 随机抽取的 1000 条音频（`crema_data_1000/metadata.csv`）。

---

## 2. 统计检验方法（组级）

### 2.1 通用检验函数

组间检验使用 `run_two_group_test()`，规则如下：

1. 输入两组样本值（自动转数值并去除无效值）
2. 若任一组样本数 `< 5`，记为无法检验（`significant=0`）
3. 正态性粗检：
   - 默认用 `Shapiro`（样本数不超过 5000）
   - 若两组都近似正态，使用 `Welch t-test`
   - 否则使用 `Mann-Whitney U`（双侧）
4. 显著性阈值：`alpha = 0.05`

> 说明：本次输出中实际检验基本走的是 `Mann-Whitney U`。

### 2.2 三个任务的组级检验设计

- 性别（`male vs female`）：
  - 特征：`f0_mean`, `spec_centroid_mean`
- 年龄（`young vs old`）：
  - 特征：`f0_mean`, `pause_ratio`
- 情绪（`one-vs-rest`）：
  - 对每个情绪标签分别做正类 vs 其余样本
  - 特征：`f0_range`, `energy_range`, `spec_flux_mean`, `mfcc_delta_std`

---

## 3. 单样本打分规则（0/1）

### 3.1 区间定义

单样本是否“符合该标签统计规律”使用组内区间判断：

- `lower = mean - 2 * std`
- `upper = mean + 2 * std`

### 3.2 `gender_sig`

`gender_sig = 1` 需要同时满足：

1. 性别主特征 `f0_mean` 组级检验显著（`p < 0.05`）
2. 样本 `f0_mean` 落在其性别组区间内（`mean ± 2*std`）

否则为 `0`。

### 3.3 `age_sig`

对 `f0_mean` 与 `pause_ratio` 分别投票：

- 仅当该特征在组级检验显著且样本值落入对应年龄组区间时，记 1 票
- 票数 `>= 1`，则 `age_sig = 1`；否则 `0`

### 3.4 `emotion_sig`

针对样本所属情绪标签（one-vs-rest）：

- 在 4 个情绪特征里，统计“该特征检验显著 + 样本落入该情绪区间”的满足数
- 满足数 `>= 2`，则 `emotion_sig = 1`；否则 `0`

### 3.5 总分与高分样本

- `sig_score = gender_sig + age_sig + emotion_sig`，范围 `0~3`
- 高分样本定义：`sig_score >= 2`

---

## 4. 本次结果摘要（来自 summary.json）

`summary.json` 内容为：

- `total_samples = 1000`
- `feature_status_counts = {"ok": 1000}`
- `task_significant_feature_count = {"gender": 2, "age": 1, "emotion": 18}`
- `sig_score_distribution = {"0": 7, "1": 107, "2": 41, "3": 845}`
- `sig_score_eq_3_count = 845`
- `sig_score_ge_2_count = 886`
- `high_score_threshold = 2`

解释：

1. `feature_status_counts["ok"]=1000` 表示 1000 条样本都完成了特征提取流程。
2. `task_significant_feature_count` 是“显著特征数”，不是“显著样本数”。
3. 情绪任务显著特征数较高（18/20），意味着在该数据集上情绪相关声学差异较明显。
4. `sig_score=3` 样本很多（845条），说明当前规则下多数样本满足三任务统计一致性。

---

## 5. 与 group_test_results.csv 的对应关系

- 性别：2个特征均显著（`f0_mean`, `spec_centroid_mean`）
- 年龄：`f0_mean` 显著，`pause_ratio` 不显著
- 情绪：本次实际出现 5 类情绪（`angry/fear/happy/neutral/sad`）
  - 每类 4 个特征，共 20 项检验
  - 其中 18 项显著

---

## 6. 结果使用建议

1. 当前分数更适合“统计一致性筛选”，不等同于分类置信度。
2. 若想让高分样本更“严格”，可收紧区间（如 `mean ± 1.5*std` 或 IQR）。
3. 若想降低同数据集内偏乐观现象，可采用训练/验证拆分后再打分。

