# CREMA-D 1000样本收紧规则结果说明（mean ± 1.5*std）

## 1. 本次重跑配置

命令：

```bash
python main.py --metadata crema_data_1000/metadata.csv --outdir crema_outputs_1000_tight_std15 --interval-method std --std-multiplier 1.5
```

关键配置：

- `interval_method = std`
- `std_multiplier = 1.5`（比之前 `2.0` 更严格）
- `iqr_multiplier = 1.5`（本次未使用，因为 interval_method=std）
- `alpha = 0.05`

---

## 2. 统计方法与打分规则（与上次一致，仅区间更紧）

### 2.1 组级检验

- 先做组级检验，不做逐条样本检验
- 正态性粗检后自动选择：
  - `Welch t-test`
  - 或 `Mann-Whitney U`
- 显著性判定：`p < 0.05`

### 2.2 单样本打分

- 区间判定从 `mean ± 2*std` 收紧为 `mean ± 1.5*std`
- `gender_sig`：`f0_mean` 组级显著 + 样本落入对应性别区间
- `age_sig`：`f0_mean` / `pause_ratio` 中满足“显著 + 区间内”的特征票数 >= 1
- `emotion_sig`：所属情绪 one-vs-rest 中满足“显著 + 区间内”的特征数 >= 2
- `sig_score = gender_sig + age_sig + emotion_sig`
- 高分样本阈值：`sig_score >= 2`

---

## 3. 新结果（summary.json）

对应文件：

- `crema_outputs_1000_tight_std15/summary.json`

摘要：

- `total_samples = 1000`
- `feature_status_counts = {"ok": 1000}`
- `task_significant_feature_count = {"gender": 2, "age": 1, "emotion": 18}`
- `sig_score_distribution = {"0": 21, "1": 130, "2": 74, "3": 775}`
- `sig_score_eq_3_count = 775`
- `sig_score_ge_2_count = 849`

---

## 4. 与旧结果（mean ± 2*std）对比

旧结果目录：`crema_outputs_1000`  
新结果目录：`crema_outputs_1000_tight_std15`

主要变化：

- `sig_score == 3`：`845 -> 775`（减少 70）
- `sig_score >= 2`：`886 -> 849`（减少 37）
- `sig_score == 0`：`7 -> 21`（增加 14）
- `sig_score == 1`：`107 -> 130`（增加 23）

三维通过率变化（均值）：

- `gender_sig`：`0.866 -> 0.828`
- `age_sig`：`0.875 -> 0.820`
- `emotion_sig`：`0.983 -> 0.955`
- `sig_score` 平均值：`2.724 -> 2.603`

结论：规则收紧后，高分样本比例下降，筛选力度增强，结果更有区分度。

