# Stage B1 实验报告：Selective vs Full Training Baseline

> **分析日期**：2026-04-07（基于 2026-04-01 训练数据，分析脚本重构后重新运行）
> **实验目标**：完成 `next_steps_experiment_analysis_plan.md` 中 Stage B1 对照（标准 full training baseline）
> **对比对象**：
> - `selective`：`phase3_outputs/tensorboard`（delta-based selective training）
> - `full_phase3`：`phase3_full2_outputs/tensorboard`（`phase3.train --disable_mask`，全 speech token 参与 loss）
> **分析脚本**：`stage_b_analysis/run_analysis.py`
> **图表目录**：`stage_b_analysis/figures/`
> **结构化输出**：`stage_b_analysis/stage_b_summary.json`
> **数据来源**：`stage_b_analysis/*.csv`（TensorBoard 导出）

---

## 关键发现（Executive Summary）

| 维度 | 结果 | 数值 |
|------|------|------|
| Train loss | Selective 略低 | 1.7610 vs 1.7870（差 0.026） |
| CV loss | Full 略低 | 4.0850 vs 4.1147（差 0.030） |
| 每步耗时 | Full 更快 | 1.20 s/step vs 1.40 s/step |
| CV 下降效率 | 近似相当 | 3.36 vs 3.42 loss/ks |
| 收敛稳定性 | 两者均稳定收敛 | — |

**总结**：在当前小规模（807 样本 / 3 epoch / 99 步）配置下，selective training 与 full training 表现非常接近。Selective 的 train loss 更低（更强训练拟合），但 CV loss 略高（泛化略弱）。差异均在 0.03 以内，尚不构成统计上有力的结论。需要 B2（random mask）和 B3（random reference）来进一步区分。

---

## 1) 实验设置与对照公平性

本次 B1 对照保证了以下一致性：

| 设置项 | 值 | 备注 |
|--------|-----|------|
| 训练集 | `phase2_outputs/data/train.data.list` | 同一份 |
| 验证集 | `phase1_outputs/data/cv.data.list` | 同一份 |
| 初始化 | `phase1_outputs/rm/rm_frozen.pt` | 同一检查点 |
| Epoch | 3 | — |
| 学习率 | 5e-6 | constantlr |
| accum_grad | 2 | — |
| 代码骨架 | `phase3.train` | 同一套 writer/tag |
| **唯一区别** | mask 启用/禁用 | selective 使用 Phase 2 mask；full 使用 `--disable_mask` |

因此，两组实验的差异可以完全归因于 token selection 策略。

---

## 2) 详细指标对比

### 2.1 Train Loss 曲线

| 模型 | 初始 loss (Step 1) | 最终 loss (Step 99) | 下降幅度 |
|------|-------------------|--------------------:|---------|
| Full | 2.2929 | 1.7870 | 0.5058 |
| Selective | 2.3171 | 1.7610 | **0.5561** |

- 两条曲线都稳定下降，无异常震荡。
- Selective 的 train loss 下降幅度略大（+0.050），最终值略低。
- **解释**：selective 只在高 delta token 上计算 loss / 反传，天然偏向”难 token”拟合，因此 train loss（在 selected token 上衡量）更容易快速下降。

### 2.2 CV Loss 曲线（step 对齐）

| Step | Full CV | Selective CV | 差值 (Full - Sel) |
|-----:|--------:|------------:|------------------:|
| 34   | 4.3227  | 4.3852      | -0.0625           |
| 67   | 4.2169  | 4.2686      | -0.0517           |
| 100  | 4.0850  | 4.1147      | **-0.0297**       |

- 在所有三个对齐检查点上，Full 的 CV loss 均更低。
- 差距随训练推进逐步缩小（0.063 → 0.052 → 0.030），**趋势表明若训练更长，selective 可能追上甚至反超**。

### 2.3 Wall-Clock 分析

| 指标 | Full | Selective |
|------|------|-----------|
| 总 wall-clock (98 steps) | 117.8 s | 137.6 s |
| 每步耗时 | 1.20 s | 1.40 s |
| Selective / Full 比 | — | 1.168× |
| CV loss 下降速率 (loss/ks) | 3.36 | 3.42 |

- Full 在 step-time 上更快（约快 17%）。
- 但 CV loss 下降效率（每千秒下降量）两者接近。
- **注意**：selective 每步更慢的原因可能是 mask 索引操作的额外开销，在当前小 batch 下比例较大。

### 2.4 最终 Checkpoint 验证

| 模型 | Final CV Loss |
|------|--------------|
| Full | **4.0850** |
| Selective | 4.1147 |
| 差值 | 0.0297 |

Full baseline 在最终 CV loss 上小幅领先（约 0.03）。

---

## 3) 可视化（见 `b1_*.png`）

| 图表 | 内容 |
|------|------|
| `b1_loss_curves.png` | Train loss（smoothed）+ CV loss 并排对比 |
| `b1_cv_comparison.png` | CV loss 柱状图（各 checkpoint 对齐） |
| `b1_wallclock_train_loss.png` | Wall-clock 时间轴上的 train loss 对比 |
| `b1_final_comparison.png` | 最终 train / CV loss 柱状对比 |

---

## 4) B1 结论

在当前数据规模和训练配置下：

1. **Selective 与 Full 都能稳定收敛**，无训练不稳定问题。
2. **Selective 在 train loss 上略占优势**（更强训练拟合），符合”聚焦难 token”的预期。
3. **Full 在 CV loss 上持续小幅领先**，且 wall-clock 更快。
4. **差距很小**（CV loss 差仅 0.03），且 CV loss gap 在缩小，不排除训练更长后 selective 追上的可能。
5. **结论**：本轮 B1 暂不支持”selective 在泛化上优于 full baseline”的强结论，但也不否定 selective 路线的价值。

---

## 5) 局限性与注意事项

- **规模极小**：仅 807 样本、3 epoch、99 步。在如此小的训练量下，selective 的优势可能尚未充分体现。
- **CV loss 指标的可比性**：selective 的 train loss 只在 selected token 上衡量，而 CV loss 在全部 token 上衡量，这可能导致 train/cv 之间的关系与 full training 不同。
- **无生成质量指标**：当前仅比较了 loss 曲线，未涉及最终 TTS 生成质量（需 Stage D 补充）。

---

## 6) 对 Stage B2 / B3 的启发

| 后续实验 | 要回答的问题 | 与 B1 的关系 |
|----------|-------------|-------------|
| B2: Random mask baseline | Selective 的优势（train loss 更低）是否来自”按 delta 选 token”，还是仅来自”稀疏反传” | 如果 random mask 也能达到类似 train loss，说明 delta selection 的独特价值有限 |
| B3: Random-reference baseline | 当前 delta 信号质量是否足够高 | 如果换随机 reference 后 selective 表现明显下降，说明高质量 reference 确实关键 |

建议在 B2 / B3 中沿用本报告的同一比较模板（train/cv 曲线、step 对齐、wall-clock 对齐、最终 CV loss），保证跨实验可比性。
