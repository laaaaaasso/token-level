# 后续实验与分析计划（分阶段）

## 文档目的

本文件用于明确：在 phase0–phase3 已经跑通之后，下一步应补哪些**分析**与**对照实验**，以便逐步验证项目的核心猜想，而不是继续无控制地扩展工程。

---

## 当前阶段结论

目前已经完成的是：

- reference model 训练
- token-level scoring / masking
- selective backprop 训练闭环

这说明方法在工程上已经可落地。  
但从研究验证角度看，目前更重要的是补齐：

1. **Phase 2 打分是否真的有信息**
2. **Selective training 是否优于普通 full training**
3. **优势是否来自“有意义的 token 选择”，而不是随机稀疏训练**
4. **优势是否来自“高质量 reference”，而不是任意一个 reference model**
5. **在更大规模数据上结论是否仍然成立**

---

# Stage A：先做合法性与可解释性分析（最高优先级）

## A1. 检查 Phase 2 的 delta 是否真的有意义

### 目标
确认 `L_delta = L_target - L_ref` 不是数值噪声，而是真正有区分度的 token-level 信号。

### 必做分析
对 `token_scores.pt` 做以下统计：

- 全体 token 的 `delta` 均值、标准差、分位数
- `target_loss` 与 `ref_loss` 的均值、标准差
- `|delta| < 1e-6`、`|delta| < 1e-5`、`|delta| < 1e-4` 的 token 比例
- `delta` 直方图
- `target_loss` vs `ref_loss` 散点图
- 每条 utterance 的 `delta` 均值 / 方差分布

### 为什么这一步最重要
如果 `target_loss ≈ ref_loss`，那么 top-k mask 可能只是按微小数值噪声排序，后面的 selective training 就缺乏科学解释基础。

### 建议输出
形成一份小报告，至少包含：

- 统计表
- 直方图
- 散点图
- 简短结论：当前 delta 是否具有实际区分度

---

## A2. 检查 selected token 是否具有结构性

### 目标
判断被选中的 token 是否不是随机散落，而是呈现结构性聚集。

### 必做分析
对每条样本统计：

- selected ratio
- selected token 的连续段长度分布
- selected token 是否集中在若干局部区段
- selected / unselected token 的 loss 分布差异

### 可视化建议
随机抽取若干 utterance，画出：

- 横轴：speech token index
- 纵轴：delta
- 颜色或标记：selected / unselected

### 你想看到的结果
不是完全随机散点，而是某些片段形成明显高 delta 区域。  
这更符合“某些表达片段更值得训练”的假设。

---

## A3. 把 token selection 与 CREMA-D 标签关联起来

### 目标
检验 token 选择是否与“表现力属性”有关，而不只是模型训练噪声。

### 建议分析维度
按样本标签分组比较：

- emotion
- speaker
- gender
- age group（如果标签可用）

### 可做统计
- 每组的平均 delta
- 每组的 selected ratio
- 每组内 delta 分布
- selected token 数量是否在某些情绪类别显著更高

### 价值
如果某些情绪样本稳定地产生更高比例的 high-delta token，这会直接支持“高表现力语音更容易产生高价值训练信号”的猜想。

---

# Stage B：补最关键的训练对照实验

## B1. 标准 full training baseline

### 目标
验证 selective training 是否真的优于标准训练。

### 实验设置
在与 selective training 尽可能一致的条件下，跑一个标准 baseline：

- 相同训练集
- 相同初始化
- 相同 epoch / step
- 相同 batch size
- 相同优化器和学习率
- 唯一区别：**不使用 selective mask，所有 speech token 全部参与反传**

### 需要比较的指标
- train loss 曲线
- cv loss 曲线
- 相同 step 下的 loss
- 相同 wall-clock time 下的 loss
- 最终 checkpoint 的验证表现

### 为什么必须做
没有这个 baseline，就无法回答“selective training 到底更好还是更差”。

---

## B2. Random mask baseline

### 目标
验证效果是否来自“按 delta 选 token”，而不是“只是少训了 40% token”。

### 实验设置
保持 selective training 框架不变，但把 token mask 改成随机生成：

- 随机选择 60% speech token 参与反传
- 保持选中比例与当前实验一致

### 需要比较的内容
- train/cv loss
- 收敛速度
- 最终生成质量
- 与 delta-based mask 的差距

### 判据
如果 random mask 与 delta mask 差不多，说明当前 selection rule 的有效性很弱。  
如果 delta mask 明显更优，说明 token selection 确实带来了信息增益。

---

## B3. Random-reference baseline

### 目标
验证优势是否来自“高质量 reference model”，而不是任意一个 reference。

### 实验设置
构造一组随机参考集 `D_ref_random`：

- 样本量与当前 `D_ref` 相同
- 但不经过统计筛选，直接随机采样

然后：

1. 用 `D_ref_random` 训练一个 random reference model
2. 用它做 Phase 2 scoring
3. 用对应 mask 做 selective training
4. 与当前“统计筛选 reference”路线比较

### 你要回答的问题
- 高质量 `D_ref` 产生的 delta 分布是否更有区分度？
- 高质量 `D_ref` 产生的 mask 是否更稳定？
- 最终训练收益是否更好？

### 这是验证核心猜想的关键一步
因为你的研究不只是想证明“selective training 有用”，还想证明“高质量 reference 的构造方式有意义”。

---

# Stage C：训练效率与数据效率分析

## C1. 做 wall-clock efficiency 分析

### 目标
验证 selective training 是否真的提升效率，而不是只减少了反传 token 数量。

### 建议记录
- 每 step 平均耗时
- 每 N step 累计耗时
- 每单位时间的 loss 降低速度
- 每 1000 个参与反传 token 带来的 loss 改善

### 为什么重要
你的“高表现力”定义本质上强调的是：  
在保证性能不明显下降的前提下，用更少数据 / 更少训练信号达到相近效果。

---

## C2. 用“全 token 评估”做验证集比较

### 目标
避免训练指标与评估指标不一致。

### 要求
即使 selective training 只对 top-k token 回传，验证时也应统一在：

- **全部 speech token 上**
- 使用同一评估方式

### 要比较的模型
- full training baseline
- delta-based selective
- random mask
- random-reference selective

### 关键点
不能只看 selective loss，因为 selective loss 天然只统计一部分 token，和 full training loss 不可直接比较。

---

# Stage D：生成质量分析（TTS任务必须补）

## D1. 固定测试集上的合成对比

### 目标
确认 selective training 的收益不只体现在 token loss，而是能反映到最终语音生成。

### 建议做法
固定一组测试文本和 reference 条件，比较以下模型的生成结果：

- full training baseline
- delta-based selective
- random mask
- random-reference selective

### 建议保存
每个模型生成同一批样本，方便主观对比。

---

## D2. 至少补一个客观生成指标

### 可选指标
根据你当前条件，至少补其中一部分：

- ASR-CER / WER
- speaker similarity
- emotion classifier confidence / accuracy
- duration 或韵律相关弱指标

### 理由
你的最终任务是 TTS，不是纯语言模型训练。  
如果只看 loss，不看生成结果，论文论证会很弱。

---

## D3. 小规模主观听感分析

### 建议做法
先不追求大规模 MOS，可以先做一个小样本人工对比：

- 清晰度
- 情感感知
- 自然度
- 与 reference 风格一致性

### 价值
哪怕样本量不大，也能帮助判断：  
被选中的 token 是否真的更偏向“表达性关键片段”。

---

# Stage E：再扩到更大规模数据

## E1. 何时扩到完整 CREMA-D

### 建议时机
在完成 Stage A–D 后，再扩到完整 CREMA-D。

### 原因
如果在机制尚未验证清楚前直接扩数据，计算成本会上升，但你仍然不知道自己要验证什么。

---

## E2. 为什么完整 CREMA-D 仍然值得做

### 价值
完整 CREMA-D 能带来：

- 更稳定的统计量
- 更均衡的情绪 / speaker 覆盖
- 更可信的基线对比
- 更稳的结论

### 适合回答的问题
- 当前观察到的规律是否能稳定复现
- 某些情绪类别的高 delta 现象是否仍然存在
- selective training 的收益是否在更大样本上保留

---

## E3. 对完整 CREMA-D 的预期定位

完整 CREMA-D 更适合作为：

- **正式验证集**
- **稳健性实验集**
- **论文中的主要实证部分之一**

但它仍不是最终的“真实世界高表现力语音”结论终点，因为 CREMA-D 本身仍属于受控、表演型语料。

---

# 推荐执行顺序

## 第一阶段：立刻补
1. Phase 2 delta 分布分析  
2. selected token 结构分析  
3. label 关联分析  

## 第二阶段：补主对照实验
1. full training baseline  
2. random mask baseline  
3. random-reference baseline  

## 第三阶段：补效率和质量验证
1. wall-clock efficiency  
2. 全 token 验证集评估  
3. 生成质量客观指标  
4. 小规模主观听感分析  

## 第四阶段：扩数据
1. 完整 CREMA-D  
2. 在完整数据上复现实验  
3. 比较结论是否稳定  

---

# 最终你要回答的核心研究问题

完成以上阶段后，你的项目应该能够比较扎实地回答以下问题：

1. **统计筛选得到的 `D_ref` 是否真的更有价值？**
2. **reference model 是否提供了有意义的 token-level baseline？**
3. **按 delta 选择 token 是否优于随机稀疏训练？**
4. **Selective training 是否能提高训练效率或数据效率？**
5. **这种收益是否能转化为最终 TTS 生成质量的收益？**
6. **这些结论在更大规模 CREMA-D 上是否仍然成立？**

---

# 现阶段最重要的一句话

**先不要急着无脑扩数据；先判断当前 token selection 是否真的携带有效信息。**  
如果这一步成立，再做 baseline、质量评估和完整 CREMA-D 扩展，整条研究线会更稳。
