# Stage D 分析报告：生成质量分析

> **分析日期**：2026-04-10
> **分析脚本**：`stage_d_analysis/generate_samples.py`（D1）、`stage_d_analysis/compute_metrics.py`（D2）、`stage_d_analysis/run_analysis.py`（综合分析）
> **图表目录**：`stage_d_analysis/figures/`
> **生成音频**：`stage_d_analysis/generated_audio/`（5 模型 × 12 条件 = 60 wav 文件 + 12 参考音频）
> **结构化输出**：`stage_d_analysis/stage_d_summary.json`、`stage_d_analysis/objective_metrics.json`

---

## 关键发现（Executive Summary）

| 分析 | 结论 | 核心证据 |
|------|------|----------|
| D1: 生成能力 | ✅ 所有模型均能正常生成 | 5 模型 × 12 条件，60/60 成功 |
| D2: 可懂度 (WER) | ⚠️ Selective 略差 | Selective WER=0.343 vs 其余三个=0.301 |
| D2: 说话人相似度 | ✅ Random mask 最佳 | SpkSim: Random mask (0.700) > Selective (0.692) > Full (0.675) |
| D2: 时长匹配 | ⚠️ 所有模型偏短 | 生成时长仅为参考的 50-62% |
| 模型间差异 | ⚠️ 差异微小 | 与 pretrained 对比，fine-tuning 影响有限 |

---

## D1：固定测试集合成对比

### 测试配置

选取 12 个多样化条件（4 句子 × 3 情绪），覆盖 CREMA-D 的主要情绪类型：

| 句子 | 情绪 | 参考说话人 |
|------|------|-----------|
| It's eleven o'clock | angry, happy, sad | 1008, 1017, 1046 |
| I'm on my way to the meeting | angry, happy, neutral | 1058, 1074, 1055 |
| Don't forget a jacket | angry, sad | 1065, 1049 |
| I think I have a doctor's appointment | happy, neutral | 1037, 1036 |
| That is exactly what happened | angry, sad | 1008, 1082 |

### 评估模型

| 模型 | 说明 |
|------|------|
| Pretrained | CosyVoice2-0.5B 原始权重（对照基准） |
| Selective | Delta-based token selection（核心方案） |
| Full baseline | 标准全 token 训练 |
| Random mask | 随机 60% token 训练 |
| Random ref (fair) | 随机 reference + delta mask |

### 推理方式

使用 CosyVoice2 的 `inference_cross_lingual` 模式（跨语言/zero-shot），仅替换 LLM 权重，flow + hift 保持预训练。

### 生成结果

所有 60 个样本均成功生成。平均生成时长：

| 模型 | 平均时长(s) |
|------|-------------|
| Pretrained | 1.43 |
| Selective | 1.47 |
| Full baseline | 1.43 |
| Random mask | 1.56 |
| Random ref (fair) | 1.26 |

---

## D2：客观生成指标

### 指标说明

1. **ASR-WER**：Whisper (base.en) 转写，与 ground truth 文本计算 Word Error Rate
2. **Speaker Similarity**：使用 CosyVoice2 的 campplus 说话人编码器提取 embedding，计算生成音频与参考音频的余弦相似度
3. **Duration Ratio**：生成音频时长 / 参考音频时长

### 总体结果

| 模型 | Avg WER ↓ | Avg SpkSim ↑ | Avg DurRatio |
|------|-----------|-------------|--------------|
| Random mask | **0.301** | **0.700** | 0.622 |
| Full baseline | **0.301** | 0.675 | 0.567 |
| Random ref (fair) | **0.301** | 0.686 | 0.497 |
| Selective (delta) | 0.343 | 0.692 | 0.580 |
| Pretrained | 0.301 | 0.687 | 0.561 |

### Speaker Similarity 排名（Fine-tuned 模型）

| 排名 | 模型 | Avg SpkSim |
|------|------|-----------|
| 1 ★ | Random mask | **0.700** |
| 2 | Selective (delta) | 0.692 |
| 3 | Random ref (fair) | 0.686 |
| 4 | Full baseline | 0.675 |

### 按情绪的 Speaker Similarity

| 模型 | angry | happy | neutral | sad |
|------|-------|-------|---------|-----|
| Pretrained | 0.690 | 0.699 | 0.693 | 0.667 |
| Selective | 0.690 | 0.671 | 0.721 | 0.695 |
| Full baseline | 0.645 | 0.683 | 0.728 | 0.672 |
| Random mask | **0.708** | **0.717** | 0.709 | 0.668 |
| Random ref (fair) | 0.697 | 0.707 | **0.724** | 0.624 |

### WER 分析

- 4 个模型的 WER 完全相同（0.301），说明 fine-tuning 对可懂度的影响极为有限
- **Selective 是唯一 WER 更高的模型**（0.343），主要在 angry 情绪上表现更差（0.440 vs 0.315）
- 这可能与 delta selection 偏向 high-loss token（通常是更困难/情绪化的片段）有关

### Duration Ratio 分析

- 所有模型生成的音频**显著短于参考音频**（ratio 0.50-0.62）
- 这是 cross_lingual 模式的已知特性，而非 fine-tuning 导致
- Random mask 产生最接近参考时长的结果（0.622）

---

## 综合讨论

### 1. Fine-tuning 对生成质量影响有限

Pretrained model 的 SpkSim（0.687）与所有 fine-tuned 模型相当（0.675-0.700），说明在当前小规模 fine-tuning（3 epoch, 807 样本）下，LLM 权重变化不足以对最终合成质量产生显著影响。

### 2. Random mask 在生成质量上仍然领先

与 Stage B（CV loss）和 Stage C（统一评估）的结论一致，random mask 在 SpkSim 上取得最佳结果，进一步支持"随机稀疏训练的正则化效果"假说。

### 3. Selective training 的可懂度劣势

Selective 是唯一 WER 上升的模型。这可能因为 delta-based selection 过度聚焦于 high-loss 的情绪化片段，导致模型在正常语音段的建模能力轻微退化。

### 4. 差异量级很小

- SpkSim 差异范围：0.025（0.675-0.700）
- WER 差异范围：0.042（0.301-0.343）
- 在 12 个样本上，这些差异的统计显著性有限

---

## 可视化参考

| 图表 | 文件 | 内容 |
|------|------|------|
| D2 Overview | `d2_metrics_overview.png` | SpkSim、WER、DurRatio 三指标条形图 |
| D2 Heatmap | `d2_spksim_by_emotion.png` | 按情绪的 SpkSim 热力图 |
| D2 Radar | `d2_radar.png` | 四模型综合质量雷达图 |

---

## 局限性

1. **样本量极小**：仅 12 个测试条件，统计结论不够稳健
2. **无主观评估**：WER 和 SpkSim 不能完全代替人耳感知
3. **Cross-lingual 模式**：使用的不是 CREMA-D 的 zero-shot 模式（需要 prompt text + prompt wav），可能未充分发挥 fine-tuned 模型的能力
4. **时长偏短**：所有模型的 DurRatio <0.65，说明韵律层面还有改进空间
5. **单次生成**：未做多次采样取平均，受随机种子影响
