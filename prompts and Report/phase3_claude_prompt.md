# Phase 3 开发任务说明（给 Claude Code）

## 任务背景

我正在做一个基于 **CosyVoice2 + RHO-1** 思路的正式项目。  
当前 **Phase 0、Phase 1、Phase 2 已完成**，现在进入 **Phase 3**。

请你只实现 **Phase 3**，不要提前实现后续阶段，也不要私自扩展任何本文档未要求的功能。

本项目当前总原则：

1. **不改 tokenizer**
2. **不改 CFM / vocoder**
3. **只在 CosyVoice2 的 text-speech LM 阶段做 RHO-1 风格改造**
4. 当前采用 **Phase 2 半离线打分 + Phase 3 selective backprop**
5. 编码过程中坚持 **Keep It Simple and Stupid (KISS)**
6. 严格遵守 **代码纪律**：**不要添加任何本文档没有明确要求的功能**

---

## 你现在要做的事

实现 **Phase 3：按 batch 内 top-k% speech tokens 做 selective backprop**。

目标是：

- 在正式训练过程中读取 **Phase 2** 已生成的 token score / token mask
- 对当前 batch 内可训练的 **speech tokens**，仅保留 top-k% token 的训练贡献
- 最终只对这些被选中的 token 计算并回传 text-speech LM 的训练损失

---

## 本阶段核心定义

对当前 batch 内全部可训练 speech tokens，按照 excess loss 排序，只保留 top-k%：

```text
I_k%(μ_i) = 1,  μ_i ∈ top-k% by L_Δ
            0,  otherwise
```

最终训练目标：

```text
L_SLM = -(1 / (N · k%)) Σ_i I_k%(μ_i) log P_θ(μ_i | y, μ_<i)
```

这里含义是：

- `I_k%(μ_i)` 是 Phase 2 产出的 token 选择信号
- 只有被选中的 speech token 参与当前 batch 的 SLM 损失
- 未被选中的 speech token 不参与当前 batch 的 SLM loss 回传

这一步与 RHO-1 的 selective training 保持一致，但只落在 **CosyVoice2 的 text-speech LM** 部分。

---

## Phase 3 的工程目标

你要实现的是：

1. 在训练时读取与 batch 对应的 token-level mask / score
2. 在 text-speech LM 的 loss 计算位置引入 selective mask
3. 仅对被选中的 speech token 计算有效 loss
4. 正确归一化 loss
5. 保持其余训练流程尽可能不变

本阶段的重点是 **修改训练损失的计算方式**，不是重新设计训练架构。

---

## 必须遵守的代码纪律

这是最高优先级要求。

### 1. 不要做本文档没写的功能
禁止私自加入下列内容：

- 不要重写整个训练器
- 不要顺手做 online scoring
- 不要顺手重做 Phase 2
- 不要顺手加入新的多指标打分函数
- 不要顺手加入 utterance-level prior 融合
- 不要顺手改 tokenizer / speech tokenizer
- 不要顺手改 CFM / vocoder / flow matching 部分
- 不要顺手做复杂配置重构
- 不要顺手做复杂缓存系统或数据库
- 不要添加任何本文档未要求的训练新机制

### 2. KISS
始终选择：

- 最直接的实现
- 最容易验证正确性的实现
- 最少改动现有工程的实现

### 3. 最小侵入
优先策略：

- 能在现有 loss 计算附近改，就不要大改训练框架
- 能复用现有 batch 字段，就不要重新设计 batch 协议
- 能新增小模块/小函数，就不要做大规模工程重构

### 4. 不要假设未来需求
只完成当前明确要求的 **Phase 3 selective backprop**。

---

## 本阶段建议实现范围

请把开发范围限制在以下内容：

### A. 训练阶段读取 Phase 2 输出
需要有一套简单直接的方式，让训练 batch 能拿到对应的 token mask。

可接受形式：

- 训练 dataset / dataloader 在取样时一并读出 `speech_token_mask`
- 或训练 step 中根据 `sample_id` 查到对应 mask

原则：**哪种方式与现有工程最兼容，就用哪种方式。**

### B. 在 SLM loss 处引入 selective mask
你需要修改或包装现有 text-speech LM loss 逻辑，使其支持：

```python
masked_loss = token_loss * speech_token_mask
```

然后只对 mask=1 的位置做归一化后的平均。

### C. 保持其他训练部分不变
除了 selective SLM loss 之外：

- 现有模型前向流程尽量不变
- 现有优化器逻辑尽量不变
- 现有日志系统尽量不变
- 非 text-speech LM 部分不要改

---

## 本阶段输入

你应假定当前已经具备以下前提：

1. 正式训练集 `D_train` 可访问
2. Phase 2 已经离线产出 token-level mask / score
3. 每个训练样本都可以关联到自己的 mask
4. 现有 CosyVoice2 text-speech LM 已经可以正常训练

你不需要重新实现：

- tokenizer
- reference model
- Phase 2 scoring 脚本
- 其他与本阶段无关的模块

---

## 本阶段输出

本阶段最终应实现：

### 1. selective SLM loss
训练时对每个 batch：

- 只保留 mask=1 的 speech token loss
- 非 speech token 不参与
- padding token 不参与
- mask=0 的 speech token 不参与

### 2. 正确归一化后的 batch loss
需要确保 loss 不是简单对全序列平均，而是只对 **被选中的有效 speech token** 平均。

例如：

```python
loss = (token_loss * selected_mask).sum() / selected_mask.sum()
```

注意要处理 `selected_mask.sum() == 0` 的边界情况。

### 3. 可训练主流程
训练脚本运行后，应能正常：

- 前向
- 计算 selective loss
- backward
- optimizer.step()

---

## 关于 mask 的要求

本阶段默认直接使用 **Phase 2 已产出的 token-level mask**，而不是在训练时重新排序重算。

也就是说，Phase 3 第一版默认流程是：

1. 先离线得到 mask
2. 训练时直接读取 mask
3. 用 mask 过滤 token loss

请不要把 Phase 3 改成在线 top-k 重算版本，除非本文档明确要求；当前没有这个要求。

---

## 关于 per-token loss 的要求

只关心 **speech token 部分** 的 loss。

请确保：

- text token 不参与 selective SLM loss
- padding token 不参与
- 只对 speech token 位置的 next-token CE loss 做选择
- loss 与 Phase 2 的 token 对齐方式一致

如果现有模型 forward 输出的是全序列 logits，请你：

1. 明确找到 speech token 对应位置
2. 提取这些位置的 token loss
3. 再与读取到的 `speech_token_mask` 对齐

---

## 推荐实现方式

请按最直接稳妥的方式来：

### Step 1
复用现有 dataloader / batch 结构，在 batch 中加入：

```python
speech_token_mask
```

要求它与用于计算 speech token loss 的位置一一对应。

### Step 2
写一个非常清晰的小函数，例如：

```python
compute_masked_speech_loss(token_loss, speech_token_mask) -> Tensor
```

功能：

- 输入 per-token speech loss
- 输入对应 mask
- 输出归一化后的 selective SLM loss

### Step 3
在训练 step 中接入：

```python
token_loss = compute_speech_token_losses(model, batch)
slm_loss = compute_masked_speech_loss(token_loss, speech_token_mask)
```

### Step 4
保持现有 backward / optimizer 逻辑：

```python
slm_loss.backward()
optimizer.step()
```

---

## 你需要特别注意的风险点

请重点检查以下问题：

### 1. token 对齐问题
Phase 2 的 mask 与当前训练阶段提取出来的 speech token loss 必须严格对齐。

### 2. padding / ignore_index 问题
要确保：

- padding 不参与 loss
- ignore_index 不参与 loss
- text token 不参与 selective SLM loss

### 3. 全 0 mask 问题
某些 batch 或某些样本可能出现：

```text
selected_mask.sum() == 0
```

请给出一个简单稳定的处理方式，例如：

- 跳过该 batch 的 SLM 更新
- 或返回 0 loss（但不能产生 NaN）

请选择**最稳妥、最简单**的一种，并明确写清楚。

### 4. batch 内长度不一致
不同样本长度不同的情况下，要明确：

- 哪些位置有效
- 哪些位置是 padding
- mask 如何与有效 speech token 对齐

### 5. 归一化正确性
损失必须除以 **被选中的有效 token 数量**，而不是总 token 数。

---

## 第一版不要做的事

以下内容全部不要碰：

- 不做在线动态重打分
- 不做训练中刷新 mask
- 不做 top-k 策略对比实验框架
- 不做额外 score function
- 不做与 CFM / vocoder 联动的 selective training
- 不做复杂多任务损失重构
- 不做分布式优化专项改造
- 不做可视化面板
- 不做额外实验控制台
- 不做任何超出 selective SLM loss 的功能扩展

---

## 我希望你最终给我的交付物

请完成后给我以下内容：

### 1. 你改了哪些文件
简洁列出。

### 2. 每个文件的作用
一句话说明即可。

### 3. 运行方式
给出最小可运行命令，例如：

```bash
python train.py \
  --train_manifest ... \
  --mask_dir ... \
  --other_existing_args ...
```

如果训练入口不是这个名字，就按真实工程入口给出。

### 4. mask 的读取方式说明
说明：

- 训练时如何找到对应 mask
- mask 的 shape / 语义是什么
- 如何与 speech token 对齐

### 5. 最小验证方法
告诉我如何验证：

- mask 已成功接入训练
- 只有被选中的 speech token 在贡献 loss
- text token / padding token 没有参与
- loss 归一化正确
- backward 能正常跑通

---

## 编码风格要求

请遵守：

- 代码简洁
- 变量名清晰
- 少封装，避免过度抽象
- 注释只写必要信息
- 优先可读性
- 优先工程稳定性

不要为了“以后可能扩展”而做复杂框架设计。

---

## 如果你发现工程里信息不够

处理原则如下：

1. 先基于现有 CosyVoice2 训练代码做合理且保守的假设
2. 不要直接大改架构
3. 不要自己发明新训练流程
4. 优先复用现有 batch 字段与 forward 输出
5. 如果必须二选一，请选择**更简单、更保守、更容易验证**的方案

---

## 最终目标（一句话）

请你帮我实现一个 **严格受控、只做 Phase 3 的 selective training 模块**：

- 训练时读取 Phase 2 产出的 token-level mask
- 只对被选中的 speech token 计算 SLM loss
- 只让这些 token 参与 backward
- 正确处理 padding / ignore_index / 对齐 / 归一化
- 不做任何超出本文档范围的扩展
