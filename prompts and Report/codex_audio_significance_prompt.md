# Codex任务说明：基于统计检验的音频性别/年龄/情绪显著性标注与高分样本筛选

## 任务目标

请你编写一套可直接运行的 Python 代码，完成以下任务：

1. 输入约 1000 条音频文件及其对应文本内容。
2. 提取用于验证 **性别（gender）**、**年龄（age）**、**情绪（emotion）** 的声学特征。
3. 主要使用 **Mann–Whitney U 检验** 和 **t 检验**，验证这三类标签是否与声学特征相匹配。
4. 在原始数据表中新增三列：
   - `gender_sig`
   - `age_sig`
   - `emotion_sig`

   每列只取：
   - `0` = 不显著
   - `1` = 显著
5. 统计总得分高的音频。定义：
   - `sig_score = gender_sig + age_sig + emotion_sig`
   - 得分范围为 `0~3`
   - 输出得分最高的样本清单，并按分数从高到低排序。

---

## 重要说明：统计学口径必须严谨

### 1. Mann–Whitney U 检验和 t 检验是“组间检验”，不是“单样本检验”
这两种检验本质上用于比较**两组样本**是否显著不同，不能严格地直接说“某一条音频显著/不显著”。

因此，这个任务中请采用下面这个**工程化定义**：

- 先在**整体数据集层面**验证：某类标签是否确实对应了显著的声学差异；
- 再对**单条样本**判断：该样本是否落在与其标签一致的“显著特征区间”中；
- 若一致，则把该条音频在该任务上的 `*_sig` 记为 `1`，否则记为 `0`。

也就是说：

- **检验负责证明“这个标签维度整体上是否成立”**
- **单样本打分负责证明“这条样本是否符合该标签的统计规律”**

请在代码中把这一逻辑写清楚，不要把 t 检验 / Mann–Whitney U 检验误写成“逐条音频显著性检验”。

---

## 输入数据格式要求

假设输入为一个 CSV，例如 `metadata.csv`，至少包含以下列：

- `audio_path`: 音频路径
- `text`: 对应文本内容
- `gender_label`: 性别标签（如 `male`, `female`）
- `age_label`: 年龄标签（建议先做二分类，例如 `young`, `old`；如果原始有多档年龄，请先映射）
- `emotion_label`: 情绪标签（建议先做二分类，例如某目标情绪 vs 其他，或按 one-vs-rest 分别验证）

如果某些标签目前是大模型生成的伪标签，也照样按上述字段读取。

---

## 总体实现思路

代码请拆成以下模块：

1. `load_metadata`
2. `extract_features`
3. `validate_gender`
4. `validate_age`
5. `validate_emotion`
6. `assign_significance_flags`
7. `rank_high_score_samples`
8. `save_results`

要求代码结构清晰、函数化、可复用。

---

# 一、特征提取方案

请主要使用以下 Python 库：

- `librosa`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

可选增强：
- `pyworld`：更稳健地提取 F0
- `parselmouth`：提取 HNR / Jitter / Shimmer / Formants

如果可选库不存在，代码应自动降级，不要报错退出。

---

## 1. 性别验证使用的特征

优先使用这些特征：

- `f0_mean`：基频均值
- `f0_min`
- `f0_max`
- `spec_centroid_mean`：频谱质心均值
- `spec_tilt_proxy`：谱倾斜代理（高频/低频能量差）

### 提取建议
- `librosa.pyin()`：提取 F0
- `librosa.feature.spectral_centroid()`：频谱质心
- 用 mel 频带高低区间能量差构造谱倾斜代理

### 统计验证方法
- 对 `male` vs `female`：
  - 先做正态性粗检查（可选）
  - 若分布较接近正态，用 `scipy.stats.ttest_ind`
  - 否则用 `scipy.stats.mannwhitneyu`
- 至少对 `f0_mean` 做主检验
- 可将多个特征的检验结果一起汇总

### 单条样本的 `gender_sig` 赋值规则
只有当以下两个条件同时满足时，`gender_sig = 1`：
1. 数据集层面的主特征（至少 `f0_mean`）组间检验显著，`p < 0.05`
2. 该样本的 `f0_mean` 落在其标签对应组的典型区间内

典型区间定义为：
- 该组样本在主特征上的 `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
- 或更简单：`mean ± 2*std`

推荐优先使用：
- `mean ± 2*std`

即：
- 若样本标签为 `male`，则看它的 `f0_mean` 是否落在 male 组的均值±2倍标准差区间内
- 若落在区间内且整体检验显著，则 `gender_sig=1`
- 否则 `gender_sig=0`

---

## 2. 年龄验证使用的特征

为了保持方法简洁，请先把年龄任务做成**二分类**：

- `young`
- `old`

如果原始年龄是多档，如：
- child / youth / adult / senior

请先在代码中支持一个映射函数，例如：
- `child`, `youth` -> `young`
- `adult`, `senior` -> `old`

优先使用这些特征：

- `f0_mean`
- `f0_std`
- `speech_rate_proxy`
- `pause_ratio`

可选增强：
- `hnr_mean`
- `jitter_local`
- `shimmer_local`

### 提取建议
- `librosa.pyin()`：F0
- `librosa.onset.onset_strength()` + `librosa.beat.tempo()`：语速代理
- `librosa.feature.rms()`：静音比例 / 停顿比例
- 如果装了 `parselmouth`，再提 HNR/Jitter/Shimmer

### 统计验证方法
对 `young` vs `old`：
- 对 `f0_mean` 做 t 检验或 Mann–Whitney U 检验
- 对 `pause_ratio` 也做同样检验
- 可以选定一个主特征 + 一个辅助特征：
  - 主特征：`f0_mean`
  - 辅助特征：`pause_ratio`

### 单条样本的 `age_sig` 赋值规则
只有当以下条件满足时，`age_sig = 1`：

1. 至少一个主特征的组间检验显著（建议 `f0_mean` 或 `pause_ratio` 的 p < 0.05）
2. 该样本在至少一个主特征上落入其标签组的典型区间
3. 如果主特征和辅助特征都满足，则更稳健；建议用“二选一即可显著，但双满足优先”

建议最终规则：

- 若 `f0_mean` 检验显著，且样本 `f0_mean` 落入对应组区间，记一票
- 若 `pause_ratio` 检验显著，且样本 `pause_ratio` 落入对应组区间，记一票
- 若得票 >= 1，则 `age_sig = 1`
- 否则 `age_sig = 0`

---

## 3. 情绪验证使用的特征

情绪是最复杂的部分。为了保持代码简洁，不要一次做复杂多分类统计建模。

请采用 **one-vs-rest** 思路。

即对于每个样本：
- 取该样本自己的 `emotion_label`
- 比较“该情绪组” vs “其他所有情绪组”

支持如下情绪标签：

- `neutral`
- `happy`
- `sad`
- `angry`
- `fear`
- `surprised`

优先使用这些特征：

- `f0_range`
- `energy_range`
- `spec_flux_mean`
- `mfcc_delta_std`

### 提取建议
- `f0_range`：来自 F0 最大最小值差
- `energy_range`：来自 RMS 最大最小差
- `spec_flux_mean`：`librosa.onset.onset_strength()` 或相邻帧频谱变化
- `mfcc_delta_std`：`librosa.feature.mfcc()` + `librosa.feature.delta()`

### 统计验证方法
对每一种情绪 `emo`：
- 令正类 = `emotion_label == emo`
- 负类 = `emotion_label != emo`
- 对以下特征分别做组间检验：
  - `f0_range`
  - `energy_range`
  - `spec_flux_mean`
  - `mfcc_delta_std`

可根据正态性自动选：
- `ttest_ind`
- 或 `mannwhitneyu`

### 单条样本的 `emotion_sig` 赋值规则
对该样本所属情绪组，统计有多少个特征满足：

1. 对应 one-vs-rest 检验显著，`p < 0.05`
2. 样本在该特征上落入本情绪组典型区间

建议规则：
- 满足特征数 >= 2，则 `emotion_sig = 1`
- 否则 `emotion_sig = 0`

也就是说，情绪比性别/年龄更严格一点，因为情绪波动更大。

---

# 二、具体 pipeline（必须按此实现）

## Step 1. 读取数据
读取 `metadata.csv`，至少包含：
- `audio_path`
- `text`
- `gender_label`
- `age_label`
- `emotion_label`

检查文件存在性，跳过损坏文件并记录日志。

---

## Step 2. 抽取声学特征

对每个音频提取以下字段，保存到 DataFrame：

### 基频相关
- `f0_mean`
- `f0_std`
- `f0_min`
- `f0_max`
- `f0_range`

对应工具：
- `librosa.pyin(y, fmin=50, fmax=600, sr=sr)`

### 能量相关
- `energy_mean`
- `energy_std`
- `energy_range`

对应工具：
- `librosa.feature.rms(y=y)`

### 节奏相关
- `tempo_bpm`
- `pause_ratio`

对应工具：
- `librosa.onset.onset_strength(y=y, sr=sr)`
- `librosa.beat.tempo(onset_envelope=..., sr=sr)`

### 频谱相关
- `spec_centroid_mean`
- `spec_bandwidth_mean`
- `spec_rolloff95_mean`
- `spec_flux_mean`
- `spec_tilt_proxy`

对应工具：
- `librosa.feature.spectral_centroid`
- `librosa.feature.spectral_bandwidth`
- `librosa.feature.spectral_rolloff`
- `librosa.stft`
- `librosa.power_to_db`

### MFCC 动态特征
- `mfcc_mean`
- `mfcc_std`
- `mfcc_delta_std`

对应工具：
- `librosa.feature.mfcc`
- `librosa.feature.delta`

### 可选（如果安装 parselmouth）
- `hnr_mean`
- `jitter_local`
- `shimmer_local`

---

## Step 3. 实现一个自动检验函数

请实现统一函数，例如：

```python
run_two_group_test(group_a, group_b, alpha=0.05)
```

要求：
1. 输入两组数值
2. 若样本太少（如任一组 < 5），返回无法检验
3. 先做正态性粗检查（例如 `scipy.stats.shapiro`，若样本过大可跳过）
4. 若两组都接近正态，用：
   - `scipy.stats.ttest_ind(..., equal_var=False)`
5. 否则用：
   - `scipy.stats.mannwhitneyu(..., alternative="two-sided")`
6. 返回：
   - `test_name`
   - `statistic`
   - `p_value`
   - `significant`（0/1）

---

## Step 4. 计算每个标签任务的“组级显著性”

### 性别
比较：
- male vs female

至少输出：
- `f0_mean` 的检验结果
- `spec_centroid_mean` 的检验结果

### 年龄
比较：
- young vs old

至少输出：
- `f0_mean`
- `pause_ratio`

### 情绪
对每个情绪做 one-vs-rest 比较，输出：
- `f0_range`
- `energy_range`
- `spec_flux_mean`
- `mfcc_delta_std`

请把所有检验结果保存成一个表，例如：
- `group_test_results.csv`

字段建议：
- `task`
- `label_name`
- `feature`
- `test_name`
- `statistic`
- `p_value`
- `significant`

---

## Step 5. 为每条样本赋予 0/1 显著性标记

请实现：

```python
assign_gender_sig(row, stats_cache, feature_df)
assign_age_sig(row, stats_cache, feature_df)
assign_emotion_sig(row, stats_cache, feature_df)
```

其中：

### 性别
- 看 `f0_mean` 是否为显著特征
- 若显著，检查该样本 `f0_mean` 是否落入其标签组区间
- 满足则 `gender_sig=1`

### 年龄
- 看 `f0_mean`、`pause_ratio`
- 若显著特征中，样本满足至少 1 个区间约束，则 `age_sig=1`

### 情绪
- 找该样本的 `emotion_label`
- 读取对应 one-vs-rest 的显著特征
- 若样本满足至少 2 个显著特征区间，则 `emotion_sig=1`

区间统一定义：
```python
lower = mean - 2 * std
upper = mean + 2 * std
```

建议同时支持可选参数，允许未来改成 IQR 规则。

---

## Step 6. 计算总分并筛选高分样本

新增：
- `sig_score = gender_sig + age_sig + emotion_sig`

然后：
- 按 `sig_score` 降序排序
- 若同分，可再按满足特征数量或音频时长排序
- 输出：
  - 全量结果表 `audio_significance_results.csv`
  - 高分样本表 `high_score_audios.csv`

高分样本建议定义为：
- `sig_score >= 2`

并统计：
- `sig_score == 3` 的样本数
- `sig_score >= 2` 的样本数
- 各分数段分布直方图

---

# 三、输出文件要求

代码运行后，至少输出以下文件：

1. `features.csv`
   - 每条音频的所有提取特征

2. `group_test_results.csv`
   - 所有组间显著性检验结果

3. `audio_significance_results.csv`
   - 原始数据 + 提取特征 + 三个 0/1 标志 + `sig_score`

4. `high_score_audios.csv`
   - 按 `sig_score` 排序的高分样本

5. `summary.json`
   - 保存整体统计摘要，例如：
   - 总样本数
   - 各任务显著特征数
   - `sig_score` 分布
   - 高分样本数量

6. 可视化图（png）
   - `gender_f0_boxplot.png`
   - `age_feature_boxplot.png`
   - `emotion_feature_boxplot.png`
   - `sig_score_hist.png`

---

# 四、代码风格要求

请按以下要求写代码：

1. 使用 Python 3.10+
2. 模块化设计，主入口为：
```python
python main.py --metadata metadata.csv --outdir outputs
```
3. 使用 `argparse`
4. 使用 `logging`
5. 所有函数写清楚 docstring
6. 代码要有异常处理，单个文件失败不能导致整体中断
7. 尽量不要写成 notebook，要写成标准工程脚本
8. 可新建如下文件结构：

```text
project/
├─ main.py
├─ feature_extraction.py
├─ stats_validation.py
├─ scoring.py
├─ utils.py
├─ requirements.txt
└─ README.md
```

---

# 五、实现细节建议

## 1. 关于文本内容
本任务输入里虽然有文本，但本轮验证的核心是**声学特征与标签的统计一致性**，因此文本内容先只保留在结果表中，不需要参与 Mann–Whitney U 或 t 检验。

## 2. 关于情绪标签
情绪建议采用 one-vs-rest，而不是一次性多分类检验，因为这样更容易把单条样本映射为 0/1 显著性标记。

## 3. 关于年龄标签
如果年龄组别太细，先映射成二分类以保证统计稳定性。

## 4. 关于缺失值
若某个音频无法提取某个特征，则填 NaN。
- 组级检验时自动 dropna
- 单样本打分时，若关键特征缺失，则该维度记 0

## 5. 关于“统计得分高的音频”
这里的“高分”不是分类概率，而是：
- 在三个标签维度上，有多少维的统计规律被该样本满足

所以要明确：
```python
sig_score = gender_sig + age_sig + emotion_sig
```

---

# 六、希望 Codex 最终交付的内容

请输出完整可运行工程，包括：

1. 全部 Python 源代码
2. `requirements.txt`
3. `README.md`
4. 一个最小可运行示例
5. 若可能，增加一个命令：
```bash
python main.py --metadata metadata.csv --outdir outputs
```
运行后直接生成全部结果文件

---

# 七、额外提醒（非常重要）

请不要把代码写成“直接根据单条音频做 t 检验/ Mann–Whitney U 检验”，这是统计上不成立的。

必须遵循以下逻辑：

- **先做组间显著性检验**
- **再做单条样本是否符合该组统计规律的判断**
- **最后把该样本在该任务上的显著性映射为 0/1**

这是本任务最关键的统计学要求。

---

# 八、建议的最小实现版本

如果时间有限，请先完成 MVP：

- 性别：
  - 用 `f0_mean`
- 年龄：
  - 用 `f0_mean` + `pause_ratio`
- 情绪：
  - 用 `f0_range` + `energy_range` + `spec_flux_mean`

并完成：
- 组间检验
- 三个 0/1 标志
- `sig_score`
- 高分样本导出

之后再逐步加入：
- `mfcc_delta_std`
- `spec_centroid_mean`
- `hnr/jitter/shimmer`
