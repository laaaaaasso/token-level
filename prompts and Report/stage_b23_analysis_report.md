# Stage B2 + B3 分析报告：Random Mask 与 Random Reference 对照实验

> **分析日期**：2026-04-08
> **实验基础**：Stage B1 已确认 selective 与 full training 在当前规模下差异很小（CV loss 差 0.03）
> **B2 目标**：验证 selective training 的优势是否来自"按 delta 选 token"，还是仅来自"稀疏反传"
> **B3 目标**：验证优势是否来自"高质量 reference model"，还是任意一个 reference 均可
> **分析脚本**：`stage_b_analysis/run_b23_analysis.py`
> **图表目录**：`stage_b_analysis/figures/`
> **结构化输出**：`stage_b_analysis/stage_b23_summary.json`

---

## 关键发现（Executive Summary）

| 对照实验 | 结论 | 核心证据 |
|----------|------|----------|
| B2: delta mask vs random mask | ❌ Delta mask **未**展现优势 | Random mask CV = **3.999** vs Selective CV = 4.115（差 0.116） |
| B3: curated ref vs random ref | ⚠️ 两者非常接近 | Fair comparison: Random ref CV = **4.105** vs Selective CV = 4.115（差 仅 0.010） |
| 覆盖率混淆变量 | ✅ 已识别并修正 | 1000-utt random ref (3.734) vs 807-utt fair version (4.105)：差距 0.371 几乎全由覆盖率差异解释 |

**总结**：在当前小规模（807 样本 / 3 epoch / 99 步）配置下：
1. **Delta-based token selection 不优于随机选择**——甚至稍差于 random mask baseline
2. **Curated reference 与 random reference 几乎无差别**——B3 fair 差异仅 0.01
3. **Random mask 是所有 fair 对照中表现最好的**——这提示 60% 稀疏训练本身可能有正则化效果
4. 原始 B3 中观察到的巨大差距（0.38）被证实主要来自 mask 覆盖率混淆变量（193 多出的 utt），而非 reference 质量差异

---

## 实验设置

### 五组实验总览

| 名称 | Mask 类型 | Reference Model | Mask 覆盖 | 唯一区别 |
|------|-----------|-----------------|-----------|----------|
| **Selective** | delta top-60% | 统计筛选 D_ref (849→807 train) | 807 utts | 原方案 |
| **Full baseline** | 无 mask（全 token） | — | — | B1 baseline |
| **Random mask** | 随机 60% | — | 807 utts | B2: 随机选 token |
| **Random ref** | delta top-60% | 随机采样 D_ref (849→807 train) | **1000 utts** | B3: 不公平版 |
| **Random ref (fair)** | delta top-60% | 随机采样 D_ref | **807 utts** | B3: 公平版 |

### 控制变量一致性

所有实验共享：
- 同一训练集（`phase2_outputs/data/train.data.list`）
- 同一验证集（`phase1_outputs/data/cv.data.list`）
- 同一初始化检查点（`phase1_outputs/rm/rm_frozen.pt`）
- 同一超参（lr=5e-6, epoch=3, accum_grad=2, constantlr）
- 同一代码框架（`phase3.train`）

---

## B2 分析：Delta-based Selection vs Random Selection

### B2 核心问题
> 效果是否来自"按 delta 选 token"，而不是"只是少训了 40% token"？

### B2 结果

| 指标 | Selective | Random Mask | 差值 (Sel − Rnd) |
|------|-----------|-------------|-------------------|
| Final Train Loss | 1.7610 | 1.7516 | +0.0094 |
| Final CV Loss | **4.1147** | **3.9992** | **+0.1155** |
| Per-step time | 1.40 s | 1.46 s | −0.05 s |

### B2 CV Loss 逐 Epoch

| Epoch | Selective | Random Mask | Gap |
|-------|-----------|-------------|-----|
| 0 | 4.385 | 4.297 | +0.088 |
| 1 | 4.269 | 4.166 | +0.103 |
| 2 | 4.115 | 3.999 | +0.116 |

### B2 解读

**Random mask 在 train loss 和 CV loss 上均优于 selective**，gap 随训练推进逐步扩大（0.088 → 0.116）。

这意味着：
- 在当前规模下，**delta-based token selection 不仅没有带来收益，反而可能有害**
- 60% 稀疏训练本身可能提供了一种**隐式正则化**效果（random mask CV = 3.999 < full baseline CV = 4.085）
- Delta signal 选出的 token 可能过度聚焦于"噪声难点"，导致模型学习偏向困难但不具代表性的 pattern

### B2 可视化

参见 `b2_selective_vs_random_mask.png`

---

## B3 分析：Curated Reference vs Random Reference

### B3 核心问题
> 优势是否来自"高质量 reference model"，还是任意一个 reference 均可？

### B3 实验过程

1. **构建 D_ref_random**：从 1000 样本中随机采样 849 个（与 D_ref 等大）
   - 与原 D_ref 重合率：84.7%（符合 849/1000 的随机预期）
2. **训练 random reference model**：同 Phase 1 配置（3 epoch, lr=1e-5）
3. **Phase 2 scoring**：用 random ref model 计算 delta
4. **Phase 3 training**：用 random-ref-based masks 训练

### B3 覆盖率混淆变量 ⚠️

在首次 B3 实验中发现严重混淆变量：
- 原始 Phase 2 scoring 仅成功评分了 **807 / 1000** 样本
- 新的 random-ref scoring 成功评分了 **1000 / 1000** 样本
- Phase 3 中未被评分的样本会**回退到 full training**（所有 token 参与）

因此，B3 的"unfair" version 拥有更完整的 mask 覆盖（1000 vs 807），不是公平对照。

**修正措施**：创建了 "fair" 版本——将 random-ref masks 过滤至与 selective 相同的 807 个 utt。

### B3 结果（Fair Comparison）

| 指标 | Selective | Random Ref (fair) | 差值 |
|------|-----------|-------------------|------|
| Final Train Loss | 1.7610 | 1.7467 | +0.0143 |
| Final CV Loss | **4.1147** | **4.1047** | **+0.0100** |

### B3 CV Loss 逐 Epoch

| Epoch | Selective | Random Ref (fair) | Gap |
|-------|-----------|-------------------|-----|
| 0 | 4.385 | 4.381 | +0.005 |
| 1 | 4.269 | 4.261 | +0.007 |
| 2 | 4.115 | 4.105 | +0.010 |

### B3 覆盖率效应量化

| 版本 | Final CV | 与 Selective 的差距 |
|------|----------|---------------------|
| Random ref (1000 utts) | 3.734 | −0.381 |
| Random ref fair (807 utts) | 4.105 | −0.010 |
| **覆盖率效应** | | **0.371** |

1000-utt 版本的 0.381 CV 优势中，**0.371（97.4%）来自覆盖率差异**，仅 0.010 可归因于 reference 质量差异。

### B3 解读

**在公平对照下，curated reference 与 random reference 几乎无差别**（CV 差仅 0.01）。

这意味着：
- 在当前规模下，**统计筛选 D_ref 的策略未产生有意义的信号质量差异**
- 两个 reference model 产生的 delta 分布差异（原 mean=1.47 vs random mean=1.40）不足以转化为训练收益
- 但覆盖率分析揭示：**确保所有训练样本都有 mask 是重要的**（193 个无 mask 样本回退为 full training 反而降低了性能）

### B3 可视化

参见 `b3_selective_vs_random_ref_fair.png` 和 `b3_confound_coverage.png`

---

## 全局排名（Fair, 807-utt）

| 排名 | 模型 | Final CV Loss | 与最佳的差距 |
|------|------|---------------|-------------|
| 1 ★ | **Random mask** | **3.9992** | — |
| 2 | Full baseline | 4.0850 | +0.086 |
| 3 | Random ref (fair) | 4.1047 | +0.106 |
| 4 | Selective (delta, curated ref) | 4.1147 | +0.116 |

Selective（当前核心方案）排名最末。

---

## 关键观察与讨论

### 1. 稀疏训练的隐式正则化
Random mask (3.999) < Full baseline (4.085)，差距 0.086。这表明在小规模数据上，**随机稀疏反传可能起到 dropout-like 的正则化作用**，比训练所有 token 更好。

### 2. Delta selection 可能引入偏置
Selective (4.115) > Random mask (3.999)，差距 0.116。Delta signal 选出的 token 偏向 high-loss 片段（Stage A 已确认 Cohen's d = 4.61）。过度聚焦这些"难 token"在数据不足时可能导致**过拟合到困难但非代表性的 pattern**。

### 3. Reference 质量在小规模下不关键
Selective (4.115) ≈ Random ref fair (4.105)，差距仅 0.010。这是因为：
- D_ref 与 D_ref_random 有 84.7% 重合（849/1000 采样导致）
- 两个 reference model 训练数据差异很小
- 在小训练量（3 epoch）下，微弱的 delta 质量差异未来得及放大

### 4. 覆盖率比选择策略更重要
B3 confound 分析显示，193 个无 mask 样本回退到 full training 造成了 0.371 的 CV 劣化。这暗示**一致性的 mask 策略**比精确的 token 选择更重要。

---

## 局限性

1. **规模极小**：807 样本、3 epoch、99 步。当前结论可能在更大规模上翻转。
2. **D_ref 与 D_ref_random 高度重叠**（84.7%）：未能充分测试 reference 质量差异。理想情况下应使用完全不重叠的对照。
3. **无生成质量指标**：仅比较了 loss 曲线。实际 TTS 生成质量可能展现不同 pattern。
4. **CV 集很小**（42 样本）：CV loss 的统计不确定性较大。
5. **单次实验**：未做多种子重复，结果可能存在随机波动。

---

## 面向后续 Stage 的建议

### 短期（Stage C/D 前）
1. **修复覆盖率问题**：确保所有训练样本都被 Phase 2 评分。调查原始 scoring 为何丢失 193 个样本。
2. **增加训练长度**：尝试 10-20 epoch，观察 delta mask 是否在更长训练后展现优势。
3. **尝试更高 topk_ratio**（如 0.8）或更低（如 0.4），观察稀疏度-性能权衡。

### 中期
4. **在更大数据集上验证**（完整 CREMA-D）：当前 1000 样本可能不足以展现 selective training 的优势。
5. **构建真正无重叠的 D_ref_random**：使用完全独立的数据源作为 random reference。
6. **引入生成质量指标**（Stage D）：loss 曲线不足以做最终判断。

### 核心研究问题的当前回答状态

| 问题 | 状态 | 证据 |
|------|------|------|
| 统计筛选的 D_ref 更有价值？ | ❌ 当前不支持 | B3: 差异仅 0.01 |
| Reference model 提供有意义的 token-level baseline？ | ⚠️ 有信号但转化弱 | A1: delta 有区分度；B2: 但不优于随机 |
| 按 delta 选 token 优于随机稀疏？ | ❌ 当前不支持 | B2: random mask 更好 0.116 |
| Selective training 提高训练效率？ | ❌ 当前不支持 | B1: 每步更慢，CV 更差 |

---

## 总结

Stage B2 和 B3 的实验结果对当前 selective training 方案提出了挑战：

**积极面**：Phase 2 的 delta signal 确实具有统计意义（Stage A 已证实），且稀疏训练本身可能有正则化价值。

**消极面**：在当前小规模配置下，delta-based selection 不优于 random selection，curated reference 不优于 random reference。核心方案（selective + curated ref）在四组 fair 对照中排名最末。

**后续方向**：增加训练规模（数据量和 epoch 数）是最重要的下一步。当前结论不应被视为对整体研究路线的否定，而是提示需要更大规模实验来验证假设。
