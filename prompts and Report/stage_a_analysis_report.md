# Stage A 分析报告：Token-Level Scoring 合法性与可解释性检验

> **分析日期**：2026-04-07（基于 2026-03-31 数据，脚本重构后重新运行）
> **数据来源**：`phase2_outputs/token_scores.pt`（807 条 CREMA-D 样本，51,963 个 speech token）
> **Phase 2 配置**：`target_ckpt = pretrained_models/CosyVoice2-0.5B/llm.pt`，`ref_ckpt = phase1_outputs/rm/rm_frozen.pt`，`topk_ratio = 0.6`
> **分析脚本**：`stage_a_analysis/run_analysis.py`
> **图表目录**：`stage_a_analysis/figures/`
> **结构化输出**：`stage_a_analysis/stage_a_summary.json`

---

## 关键发现（Executive Summary）

| 检查项 | 结论 | 核心证据 |
|--------|------|----------|
| A1: Phase 2 delta 是否有意义 | ✅ 是 | delta_std = 2.06，one-sample t-test p ≈ 0，Cohen's d = 0.71，近零 token < 0.1% |
| A2: Selected token 是否具有结构性 | ✅ 是 | run length 均值 3.01 > 随机期望 2.50（Mann-Whitney p = 2.4e-14），loss gap = +2.73（paired t p ≈ 0，Cohen's d = 4.61） |
| A3: Token selection 与标签是否有关 | ⚠️ 有统计显著但效应温和的差异 | 情绪维度 ANOVA F = 28.08, p = 7.4e-22；性别维度 t-test p = 0.90（无差异）；selected_ratio 各组均 ≈ 0.599 |

**总结**：Phase 2 的 token-level delta 是有实质区分度的信号（非噪声），选中 token 具有结构性聚集并明显偏向高 loss 片段，各标签维度上的 delta 差异更多反映”声学难度差异”而非单一标签偏置。Stage A 的三个合法性检查均通过，可以进入 Stage B 对照实验。

---

## A1：Delta 分布合法性分析

### 1.1 全体 token 统计

基于 `token_scores.pt` 中 51,963 个 speech token：

| 指标 | 值 |
|------|-----|
| Total speech tokens | 51,963 |
| Delta 均值 | 1.4702 |
| Delta 标准差 | 2.0605 |
| Delta 最小值 | -5.8752 |
| Delta 最大值 | 16.6629 |
| Target loss 均值 | 5.5681 |
| Target loss 标准差 | 2.8483 |
| Ref loss 均值 | 4.0979 |
| Ref loss 标准差 | 2.0443 |
| 精确相等 (target == ref) | 0 / 51,963 (0.0%) |

分位数：

| P01 | P05 | P25 | P50 | P75 | P95 | P99 |
|-----|-----|-----|-----|-----|-----|-----|
| -1.74 | -0.68 | 0.17 | 0.98 | 2.28 | 5.04 | 9.81 |

近零 token 比例：

| 阈值 | 比例 |
|------|------|
| \|δ\| < 1e-6 | 0.00% |
| \|δ\| < 1e-3 | 0.07% |
| \|δ\| < 1e-2 | 0.68% |

### 1.2 统计检验

| 检验 | 统计量 | p 值 | 效应量 |
|------|--------|------|--------|
| Paired t-test (target vs ref loss) | t = 162.64 | p ≈ 0 | Cohen's d = 0.71 |
| One-sample t-test (delta ≠ 0) | t = 162.64 | p ≈ 0 | — |

- delta 的分布覆盖 [-5.88, 16.66]，并非围绕 0 的微小浮点噪声。
- target / ref 均值差异约 1.47，Cohen's d = 0.71（medium-large effect），ref 模型整体 loss 显著更低。

### 1.3 可视化（见 `a1_*.png`）

- **Delta 直方图** (`a1_delta_histogram.png`)：主体集中在 0–4，有长尾（> 5）和负值尾部（target 比 ref 更好的 token）。整体右偏。
- **Target vs Ref 散点图** (`a1_target_vs_ref_scatter.png`)：大部分点在 y > x 区域，少量点 y < x。
- **Loss 分布叠加** (`a1_loss_distribution.png`)：target 分布比 ref 更宽、均值更高。
- **Per-utterance delta 均值/标准差** (`a1_per_utterance_delta.png`)：utterance 间存在明显的 delta 变异，非退化分布。

### 1.4 A1 小结

**Phase 2 的 delta 信号是有实质区分度的：** 近零 token 极少，配对检验高度显著，效应量中等偏大。可以支撑后续 selective training 与 baseline 对照。

---

## A2：Selected Token 结构性分析

### 2.1 选中比例

| 指标 | 值 |
|------|-----|
| 平均选中比例 | 0.5994 |
| 标准差 | 0.0043 |
| 范围 | [0.5909, 0.6098] |

与 top-k = 60% 设定一致。

### 2.2 连续选中段（run）分析

| 指标 | 实测值 | 随机期望 |
|------|--------|----------|
| 总 run 段数 | 10,361 | — |
| 平均 run 长度 | **3.01** | **2.50** |
| 中位数 | 2.0 | — |
| 最大 run 长度 | 31 | — |

| 统计检验 | 值 |
|----------|-----|
| Mann-Whitney U (actual > random sim) | U = 70,111,831 |
| p 值 | **2.36e-14** |

- 实测平均连续段长度显著高于随机预期（p < 1e-13）。
- 直方图对比（`a2_run_length_distribution.png`）显示 actual 分布在长 run (5+) 处有更多质量。
- **解释**：高 delta token 倾向于以 cluster 形式出现，与”局部声学难度集中”的直觉一致。

### 2.3 Selected vs Unselected Token Loss

| 类别 | 平均 Target Loss | 统计检验 |
|------|-----------------|----------|
| Selected | 6.6938 | paired t: t = 105.97, p ≈ 0 |
| Unselected | 3.9629 | Cohen's d = **4.61** |
| **差异** | **+2.7309** | |

选中 token 的 loss 显著高于未选中 token，效应量极大（d = 4.61），跨全部 807 条 utterance 一致。

### 2.4 可视化（见 `a2_*.png`）

- **Example utterances** (`a2_example_utterances.png`)：delta 值按 token 位置画出，选中/未选中分色；可见选中段在序列中形成局部”山峰”。
- **Box plot** (`a2_selected_vs_unselected_loss.png`)：两组 loss 分布完全分离。

### 2.5 A2 小结

- **结构性**：选中 token 非完全随机散落，run length 显著高于随机预期，有 cluster 特征。
- **难度属性**：选中 token loss 显著高于未选中，效应量极大。

**Stage A 第二个问题的答案：选中 token 具有稳定的结构性聚集，且明确偏向高 loss（高潜在训练价值）片段。**

---

## A3：Token Selection 与 CREMA-D 标签关联分析

> 基于 `merged_scores_labels.csv` 中 807 条样本的 per-utterance 统计。

### 3.1 按 Emotion

| Emotion | N | mean_delta | mean_selected_ratio | mean_target_loss | mean_ref_loss |
|---------|---|-----------|--------------------:|-----------------|--------------|
| angry   | 167 | 1.5980 | 0.5996 | 5.3799 | 3.7819 |
| fear    | 166 | 1.3763 | 0.5993 | 5.5884 | 4.2122 |
| happy   | 161 | 1.5684 | 0.5993 | 5.6845 | 4.1161 |
| neutral | 141 | 1.5486 | 0.5992 | 5.6805 | 4.1319 |
| sad     | 172 | 1.3889 | 0.5994 | 5.6775 | 4.2886 |

| 检验 | 统计量 | p 值 |
|------|--------|------|
| One-way ANOVA (delta_mean ~ emotion) | F = 28.08 | p = 7.4e-22 |
| Kruskal-Wallis | H = 103.16 | p = 2.1e-21 |

- 情绪维度上 delta 差异高度显著：angry/happy/neutral 的 delta 略高（target 更难），fear/sad 的 delta 略低（ref 也较吃力）。
- **但 selected_ratio 在各情绪间几乎恒定 ≈ 0.599**，说明 top-k 机制本身不产生数量偏置。

### 3.2 按 Gender

| Gender | N | mean_delta | mean_selected_ratio | mean_target_loss |
|--------|---|-----------|--------------------:|-----------------|
| female | 381 | 1.4920 | 0.5992 | 5.5994 |
| male   | 426 | 1.4944 | 0.5995 | 5.5996 |

- t-test: t = -0.12, p = 0.90。**性别维度上无显著差异**，selection 在性别上是中性的。

### 3.3 按 Text（句子 ID）

- 句子间 mean_delta 范围 [1.33, 1.76]，差异约 0.4。MTI / TAI 的 delta 最高，ITS / TSI 最低。
- 反映不同句子的声学模式在 target vs ref 之间存在不同程度的 gap。
- selected_ratio 恒定在 ≈ 0.599。

### 3.4 按 Age

- old (305) vs young (502) 差异很小：delta 差约 0.04，loss 差约 0.04。

### 3.5 A3 小结

- **数量维度**：selected_ratio 在所有标签维度上基本均匀（≈ 0.599），top-k 不偏置。
- **强度维度**：emotion 的 delta 组间差异高度显著（F = 28.08），但效应来自”声学难度差异”（如 angry 的 ref loss 更低 → delta 更大），而非简单的”高情感 = 高选中”。
- **性别 / 年龄**：无显著差异。
- “表现力 → 更多高价值 token” 的强因果结论仍需 Stage B 对照实验和 Stage D 生成质量验证。

---

## 总结与后续行动

### Stage A 三个问题的回答

| 问题 | 回答 | 置信度 |
|------|------|--------|
| Phase 2 delta 是否有信息 | ✅ 是 | 高（t ≈ 163, p ≈ 0, d = 0.71） |
| Selected token 是否有结构性 | ✅ 是 | 高（MWU p = 2.4e-14, loss gap d = 4.61） |
| Token selection 是否与表现力属性有关 | ⚠️ 有显著但温和的差异 | 中（emotion ANOVA p = 7.4e-22，但效应来自难度差异） |

### 面向 Stage B 的输入

1. **Mask 真实性已确认**：top-k mask 偏向高 loss token，非随机/噪声 → Phase 3 selective 可视为有意义的 delta-based training。
2. **对照实验合理性已建立**：full / random mask / random-reference baseline 均可与当前 selective 模型公平对比。
3. **辅助分析指标**：per-utterance delta / loss / selected_ratio 可用于 Stage B–D 的分组分析。
