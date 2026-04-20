# 基于 RHO-1 思路的 TTS Token 选择机制设计文档

## 1. 问题背景

当前方案希望借鉴 RHO-1 的核心思想：通过比较 **reference model** 与 **target model** 在 token 级别的损失差异，筛选出“更值得学习”的 token，并在训练中对这些 token 施加更大权重，或仅对其进行损失回传。

在文本语言模型中，RHO-1 采用的核心打分方式是 **excess loss**：

\[
\Delta \ell_i = \ell_{\text{target}, i} - \ell_{\text{ref}, i}
\]

并按该分数选择 top-k% token 进行训练。其动机是：若某 token 对 target model 来说仍明显难于 reference model，则该 token 更可能处于“可学习且有价值”的区域。RHO-1 的实验表明，这种选择性训练能显著提升数据效率与下游性能。

但在 **TTS / speech LM** 场景中，直接照搬“裸减法”存在明显局限，因此有必要设计更稳健的 token 选择与加权机制。

---

## 2. 为什么“直接减法 excess loss”不一定足够好

### 2.1 尺度敏感问题

直接使用

\[
s_i = \ell_{\text{target}, i} - \ell_{\text{ref}, i}
\]

默认假设 reference 与 target 的 loss 处在可直接比较的同一尺度上。但在实际训练中，这一假设未必成立。差异可能来自：

- 模型训练阶段不同；
- 模型 calibration 不同；
- 概率分布熵水平不同；
- token 预测尖锐度不同。

因此，同样大小的差值在不同 batch、不同 checkpoint、不同模型组合中未必具有相同含义。

### 2.2 高 loss 不等于高价值

在语音建模中，高 loss token 的来源更加复杂。它可能来自：

- 文本—语音对齐困难；
- 韵律或情感表达复杂；
- 说话人风格迁移困难；
- ASR 不稳定；
- tokenizer 离散化误差；
- 数据噪声或标注误差。

因此，“loss 高”不一定代表“高信息密度”或“高表现力”，也可能只是噪声。

### 2.3 语音 token 的目标不是单一文本预测

文本 LM 中，token loss 与建模目标基本一致；而在 TTS 中，token 的价值最终体现在：

- 内容一致性；
- speaker similarity；
- prosody naturalness；
- 情感表达；
- 指令可控性。

因此，只看 reference-target 间的 pointwise loss 差值，往往不足以完整刻画 token 的训练价值。

---

## 3. 设计目标

本设计希望构建一个适用于 TTS / speech LM 的 token 选择机制，满足以下目标：

1. **保留 RHO-1 的核心思想**：利用 reference model 提供“期望分布”或“对照分布”；
2. **避免裸减法带来的尺度问题**；
3. **降低噪声 token 被误选中的概率**；
4. **支持 batch 内排序与 top-p 筛选**；
5. **支持从硬选择逐步过渡到软加权**；
6. **为后续引入 ASR / SER / speaker reward 留出接口**。

---

## 4. 方案总览

本设计提出三层方案，按复杂度递进：

### 方案 A：标准化 excess loss
在保留 RHO-1 结构的前提下，对 target/ref loss 做标准化后再计算分数。

### 方案 B：软权重替代硬筛选
不再只保留 top-k% token，而是将分数映射为连续权重，对全部 token 加权训练。

### 方案 C：引入 reward / KL 的混合打分
将 excess loss 作为 proxy，同时引入可微 reward 与 reference 正则约束，向 CosyVoice 2/3 的 post-training 思路靠拢。

---

## 5. 方案 A：标准化 excess loss

### 5.1 设计动机

在你当前实验框架中，最容易落地的改进就是：**不要直接做裸减法，而是在 batch 内进行归一化或标准化后再比较。**

这样做的目的：

- 降低不同 batch 间 loss 分布漂移的影响；
- 降低 reference / target 尺度不同带来的偏差；
- 让 top-k 排序更稳定；
- 更适合你计划中的“每个 batch 内取前 60%”。

### 5.2 候选定义

#### 定义 A1：相对差值

\[
s_i = \frac{\ell_{\text{target}, i} - \ell_{\text{ref}, i}}{\ell_{\text{ref}, i} + \epsilon}
\]

优点：
- 减少 reference loss 绝对大小带来的影响；
- 对 reference 强弱变化更稳健。

缺点：
- 当 \(\ell_{\text{ref}, i}\) 很小时会放大波动；
- 需要合适的 \(\epsilon\)。

#### 定义 A2：batch 内 z-score 差值

设当前 batch 中 target loss 的均值和标准差分别为 \(\mu_t,\sigma_t\)，reference loss 的均值和标准差分别为 \(\mu_r,\sigma_r\)，则：

\[
z_t(i)=\frac{\ell_{\text{target}, i}-\mu_t}{\sigma_t+\epsilon}
\]

\[
z_r(i)=\frac{\ell_{\text{ref}, i}-\mu_r}{\sigma_r+\epsilon}
\]

定义分数：

\[
s_i = z_t(i) - z_r(i)
\]

优点：
- 非常适合 batch 内排序；
- 能抑制不同 batch 的尺度漂移；
- 易于解释：“该 token 相对 target 更难、相对 reference 更异常”。

缺点：
- 对 batch size 有一定要求；
- 小 batch 时统计量会更不稳定。

#### 定义 A3：margin excess loss

\[
s_i = \max(0,\ell_{\text{target}, i}-\ell_{\text{ref}, i}-\gamma)
\]

优点：
- 忽略微小差异；
- 更聚焦“显著高于 reference”的 token。

缺点：
- 需要调节 margin \(\gamma\)。

### 5.3 推荐

当前阶段最推荐使用：

\[
s_i = z_t(i) - z_r(i)
\]

即 **batch 内 z-score 标准化差值**。

理由：

- 与你现有“每 batch 排序、取前 60%”最兼容；
- 能直接缓解“delta 退化为 0 或分布很窄”的问题；
- 工程改动小，便于快速实验。

---

## 6. 方案 B：从硬选择改为软权重

### 6.1 设计动机

RHO-1 采用 top-k% 硬选择：只对高分 token 保留 loss，其余 token 直接 mask 掉。  
这种方法在文本 LM 中有效，但在 TTS 中存在一个问题：语音信号具有连续性，韵律、情感、时长等信息常呈局部连续变化。若直接硬截断，可能导致训练信号不连续。

因此，更稳健的策略是：**先做软加权，再尝试硬筛选。**

### 6.2 权重映射方式

给定 token 分数 \(s_i\)，可定义权重：

#### 方案 B1：Sigmoid 权重

\[
w_i = \sigma(\alpha s_i)
\]

其中 \(\alpha\) 控制权重分布的陡峭程度。

优点：
- 简单稳定；
- 权重范围固定在 \((0,1)\)；
- 便于从“弱选择”平滑过渡到“强选择”。

#### 方案 B2：Softmax 权重

\[
w_i = \frac{\exp(\alpha s_i)}{\sum_j \exp(\alpha s_j)}
\]

优点：
- 强调 batch 内相对重要性；
- 非常适合做相对排序。

缺点：
- 过于竞争化，少数 token 可能占据大部分权重。

#### 方案 B3：截断线性权重

\[
w_i = \min(\max(a s_i+b,0),1)
\]

优点：
- 可控性强；
- 线性、直观。

### 6.3 加权训练目标

使用 target model 的原始 loss 作为训练项：

\[
L = \sum_i w_i \ell_{\text{target}, i}
\]

也可以归一化为：

\[
L = \frac{\sum_i w_i \ell_{\text{target}, i}}{\sum_i w_i + \epsilon}
\]

### 6.4 为什么先软后硬

建议实验流程：

1. 先用软权重验证 score 是否确实有区分度；
2. 观察权重分布、loss 分布、训练稳定性；
3. 若趋势正确，再切换到 top-60% 硬筛选做对照。

这样可以避免一开始就被硬阈值扰动训练过程。

---

## 7. 方案 C：引入 reward / KL 的混合打分

### 7.1 设计动机

从更长远看，单一的 reference-target loss 差值只是一个 **proxy**。  
你真正关心的不是“token 比 reference 难多少”，而是：

- 它是否有助于提升文本可懂度？
- 是否有助于情感表达？
- 是否有助于 speaker similarity？
- 是否有助于 prosody naturalness？

因此，更合理的方案应当结合下游代理目标。

### 7.2 基础混合打分公式

定义：

\[
s_i = \alpha \cdot \mathrm{norm}(\ell_{\text{target}, i} - \ell_{\text{ref}, i})
+ \beta \cdot \mathrm{norm}(\ell_{\text{target}, i})
+ \gamma \cdot r_i
\]

其中：

- 第一项：保留 RHO-1 的 excess loss 思想；
- 第二项：避免仅靠差值导致分数失真；
- 第三项：引入 reward 信号。

### 7.3 reward 可选来源

#### 1）ASR 相关 reward
衡量 token 是否有助于最终语音内容被正确识别。

可选形式：
- differentiable ASR posterior；
- Token2Text 后验概率；
- utterance-level / chunk-level ASR confidence。

#### 2）SER 相关 reward
衡量 token 是否承载情感信息。

适合高表现力语音研究，特别是你当前研究主题中“表现力”的定义与此高度相关。

#### 3）Speaker reward
衡量 token 是否有助于说话人特征保持。

#### 4）Prosody / style reward
若后续有可用的韵律编码器、风格分类器，也可纳入统一 reward 框架。

### 7.4 KL 正则项

为了防止模型过度偏离 reference，可加入 token-level KL 约束：

\[
L_{\text{KL}} = D_{KL}(p_{\text{target}} \,\|\, p_{\text{ref}})
\]

总目标可写为：

\[
L = \sum_i w_i \ell_{\text{target}, i} + \lambda_{\text{reward}} L_{\text{reward}} + \lambda_{\text{KL}} L_{\text{KL}}
\]

这一路径与 CosyVoice 3 的 DiffRO 思路更接近：**reward 驱动 + reference 约束**，比纯差分更适合 speech generation。

---

## 8. 推荐的分阶段落地方案

### 阶段 1：最小改动验证版

目标：验证“batch 内标准化 excess loss”是否比裸减法更稳定。

采用：

\[
s_i = z_t(i) - z_r(i)
\]

然后做 batch 内 top-60% 选择：

\[
I_i = \mathbf{1}[s_i \text{ ranks in top }60\%]
\]

训练损失：

\[
L = \frac{\sum_i I_i \ell_{\text{target}, i}}{\sum_i I_i + \epsilon}
\]

#### 观察指标
- score 分布是否塌缩；
- 每 batch 被选 token 比例是否稳定；
- 训练 loss / acc 曲线是否较原方案更平稳；
- selected token 与非 selected token 的平均 loss 差异。

---

### 阶段 2：软加权版

目标：提升训练稳定性，降低硬阈值造成的离散扰动。

采用：

\[
s_i = z_t(i) - z_r(i)
\]

\[
w_i = \sigma(\alpha s_i)
\]

训练损失：

\[
L = \frac{\sum_i w_i \ell_{\text{target}, i}}{\sum_i w_i + \epsilon}
\]

#### 观察指标
- 权重分布是否有足够区分度；
- 与 top-60% 方案相比，训练是否更稳；
- 是否在小规模数据上更快收敛。

---

### 阶段 3：混合 reward 版

目标：让 token 价值与真正的 TTS 目标对齐，而不是只依赖 proxy。

采用：

\[
s_i = \alpha(z_t(i)-z_r(i)) + \beta z_t(i) + \gamma r_i
\]

\[
w_i = \sigma(\tau s_i)
\]

总损失：

\[
L = \frac{\sum_i w_i \ell_{\text{target}, i}}{\sum_i w_i + \epsilon}
+ \lambda_{\text{reward}}L_{\text{reward}}
+ \lambda_{\text{KL}}L_{\text{KL}}
\]

#### 观察指标
- 是否提升表达性相关指标；
- 是否改善 ASR / SER / speaker 相关代理任务；
- 是否出现“reward hacking”现象。

---

## 9. 推荐的主实验对照组

为了让实验结论清晰，建议至少保留以下对照组：

### Baseline 0：普通训练
不做 token 选择，全部 token 回传损失。

### Baseline 1：裸减法 + top-60%
\[
s_i = \ell_t(i) - \ell_r(i)
\]

### Exp 1：z-score 差值 + top-60%
\[
s_i = z_t(i) - z_r(i)
\]

### Exp 2：z-score 差值 + sigmoid 软权重
\[
w_i = \sigma(\alpha s_i)
\]

### Exp 3：z-score 差值 + reward 混合
\[
s_i = \alpha(z_t-z_r)+\beta z_t+\gamma r_i
\]

这样可以回答三个关键问题：

1. 裸减法是否真的不稳；
2. 标准化是否能改善选择质量；
3. 软权重和 reward 混合是否进一步提升 TTS 目标表现。

---

## 10. 当前最推荐的实施版本

综合理论合理性、工程成本和你当前项目推进节奏，**最推荐先实施的版本**是：

### 推荐版本 V1

#### 分数定义
\[
s_i = z_t(i) - z_r(i)
\]

#### 权重定义
\[
w_i = \sigma(\alpha s_i)
\]

#### 训练目标
\[
L = \frac{\sum_i w_i \ell_{\text{target}, i}}{\sum_i w_i + \epsilon}
\]

#### 可选升级
若已有 ASR reward，可进一步扩展为：

\[
L = \frac{\sum_i w_i \ell_{\text{target}, i}}{\sum_i w_i + \epsilon}
+ \lambda_{\text{ASR}}L_{\text{ASR}}
+ \lambda_{\text{KL}}D_{KL}(p_t\|p_r)
\]

### 选择该版本的理由

1. **保留 RHO-1 的核心思想**，仍然使用 reference-target 对照；
2. **避免裸减法的尺度问题**；
3. **比 top-k 更平滑稳定**；
4. **工程实现简单**，可直接接入现有 batch 内评分流程；
5. **便于后续扩展到 reward/KL 框架**。

---

## 11. 结论

对于当前 TTS / speech LM 项目，**直接使用**

\[
\ell_{\text{target}}-\ell_{\text{ref}}
\]

作为 excess loss 虽然可以作为 baseline，但并不应视为最终方案。

更合理的设计路线是：

1. **先用 batch 内标准化差值替代裸减法**；
2. **再用软加权替代硬筛选**；
3. **最终引入 reward 与 KL 正则，使 token 选择与 TTS 真正目标对齐**。

因此，建议将你的 token 选择机制从“裸减法 + top60%”升级为：

> **标准化差值 + 软权重 + 可扩展 reward/KL 约束**

这会比简单的 excess loss 更稳、更适合语音场景，也更符合后续向 CosyVoice 风格 post-training 框架演进的方向。
