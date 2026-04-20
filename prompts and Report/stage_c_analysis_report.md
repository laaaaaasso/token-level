# Stage C 分析报告：训练效率与数据效率分析

> **分析日期**：2026-04-10
> **分析脚本**：`stage_c_analysis/run_analysis.py`（效率分析）、`stage_c_analysis/unified_eval.py`（统一评估）
> **图表目录**：`stage_c_analysis/figures/`
> **结构化输出**：`stage_c_analysis/stage_c_summary.json`、`stage_c_analysis/unified_eval_results.json`

---

## 关键发现（Executive Summary）

| 分析 | 结论 | 核心证据 |
|------|------|----------|
| C1: Wall-clock 效率 | ❌ Selective training 每步更慢，无效率增益 | Selective 1.40s/step vs Full 1.20s/step（慢 17%） |
| C1: Loss-per-second | ⚠️ Selective train loss 下降率高于 full | Selective 2.20/ks vs Full 1.90/ks，但未转化为 CV 优势 |
| C2: 统一评估一致性 | ✅ TB CV 与统一评估高度一致 | 差异仅 0.001–0.003，排名不变 |
| C2: 最终排名 | ❌ Selective 在统一评估中仍排最末 | Random mask (4.001) > Full (4.086) > Random ref (4.108) > Selective (4.118) |

---

## C1：Wall-Clock 效率分析

### 每步训练时间

| 模型 | 秒/步 | 相对速度 |
|------|-------|----------|
| Full baseline | **1.202** | 1.000x（基准） |
| Random ref (fair) | 1.258 | 0.956x |
| Selective (delta) | 1.404 | 0.856x |
| Random mask | 1.455 | 0.826x |

**分析**：
- Full baseline 最快，因为没有 mask lookup 开销
- Selective 和 Random mask 方法有额外的 mask 查找和应用开销（~17-21% 慢）
- Random ref (fair) 介于中间

### 训练效率（Loss 下降率）

| 模型 | 总时间(s) | Train Loss 下降 | 下降率(/1000s) | 效率比 |
|------|-----------|----------------|---------------|--------|
| Random ref (fair) | 123.2 | 0.3129 | 2.539 | **1.339x** |
| Selective (delta) | 137.6 | 0.3024 | 2.197 | 1.159x |
| Random mask | 142.6 | 0.2829 | 1.984 | 1.046x |
| Full baseline | 117.8 | 0.2234 | 1.896 | 1.000x |

**注意**：这里的 "Loss 下降" 是 **train loss** 而非 CV loss。Selective 和 Random ref 在 train loss 上下降更快，但这**并未转化为 CV 优势**——说明更快的 train loss 下降可能反映的是过拟合，而非真正的学习效率提升。

### C1 核心结论

1. **Selective training 每步慢 17%**：mask lookup 和逐样本 mask 应用带来固定开销
2. **Train loss 下降率不能代表真实效率**：Selective 的 train loss 下降快，但 CV loss 更差
3. **在当前规模下，selective training 不提供 wall-clock 效率增益**

---

## C2：统一全 Token 评估

### 评估方法

对所有 5 组模型（selective, full_baseline, random_mask, random_ref, random_ref_fair）的每个 epoch checkpoint，在 CV 集（42 条样本，2662 个 speech token）上用 **全部 speech token** 计算 CE loss，确保指标可直接比较。

### 结果

| 模型 | Epoch 0 | Epoch 1 | Epoch 2 | Accuracy |
|------|---------|---------|---------|----------|
| Random mask | 4.294 | 4.166 | **4.001** | **15.21%** |
| Full baseline | 4.319 | 4.216 | 4.086 | 14.54% |
| Random ref (fair) | 4.379 | 4.263 | 4.108 | 14.43% |
| Selective (delta) | 4.384 | 4.270 | 4.118 | 14.24% |
| Random ref (1000 utts) | 4.509 | 4.490 | 4.478 | 12.36% |

### TensorBoard CV vs 统一评估对比

| 模型 | TB CV | 统一评估 | 差值 |
|------|-------|----------|------|
| Full baseline | 4.0850 | 4.0859 | +0.0009 |
| Random mask | 3.9992 | 4.0009 | +0.0017 |
| Selective | 4.1147 | 4.1179 | +0.0032 |
| Random ref (fair) | 4.1047 | 4.1081 | +0.0034 |

**差异极小（0.001–0.003）**，说明 TensorBoard 中报告的 CV loss 本身已经是在全部 speech token 上计算的（CV 时 mask 无法匹配 CV 样本，自动回退为 all-token 评估）。

### 排名确认

统一评估排名与 TB 排名完全一致：

| 排名 | 模型 | Unified CV Loss |
|------|------|----------------|
| 1 ★ | Random mask | **4.001** |
| 2 | Full baseline | 4.086 |
| 3 | Random ref (fair) | 4.108 |
| 4 | Selective (delta) | 4.118 |

### C2 核心结论

1. **统一评估确认了 Stage B 的排名**：Selective 在所有公平对照中排名最末
2. **TB CV 与统一评估一致**：无需担心评估指标不一致问题
3. **Random mask 的优势稳定存在**：在全 token 评估下仍然领先

---

## 综合讨论

### 为什么 Selective Training 在效率和效果上都不占优？

1. **Mask 开销**：逐样本查找和应用 mask 增加了约 17% 的时间成本
2. **过拟合倾向**：Delta-based selection 聚焦于 high-loss token，在小数据集上导致过拟合
3. **随机稀疏的正则化效果**：Random mask 表现最好，说明 60% 稀疏训练本身有类似 dropout 的正则化效果，而这种效果被 delta-based selection 的偏置所抵消

### 效率分析的局限性

1. 当前规模极小（99 步 / 3 epoch），timing 噪声可能较大
2. 未测量 GPU 利用率、显存占用等更细粒度指标
3. 更大规模训练中，mask 开销占比会降低（固定成本 vs 可变成本）

---

## 面向后续阶段的影响

### 对 Stage D（生成质量）的预期
- 考虑到 CV loss 差异不大（最大差距 0.117），生成质量差异可能同样微小
- 但 loss 差异不一定直接反映感知差异，仍值得评估

### 对 Stage E（扩规模）的建议
- **当前最有价值的下一步**：在更大数据/更长训练上重复实验
- 如果 selective training 的优势在规模增大后仍未出现，需考虑：
  - 调整 topk_ratio
  - 改进 delta signal（如动态更新 reference model）
  - 改进 mask 策略（如 soft weighting 而非 hard selection）

---

## 可视化参考

| 图表 | 文件 | 内容 |
|------|------|------|
| C1.1–C1.3 | `c1_efficiency.png` | 每步时间、效率条形图、wall-clock loss 曲线 |
| C1.4 | `c1_convergence.png` | 收敛速度（best-loss-so-far vs time） |
| C2.1–C2.2 | `c2_unified_eval.png` | 统一 CV loss 曲线、最终排名 |
| C2.3 | `c2_accuracy.png` | 统一 CV accuracy 曲线 |
| C2.4 | `c2_tb_vs_unified.png` | TB CV vs 统一评估对比 |
