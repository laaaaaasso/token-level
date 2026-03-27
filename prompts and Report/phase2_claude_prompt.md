# Phase 2 开发任务说明（给 Claude Code）

## 项目上下文

当前项目基于 **CosyVoice2 + RHO-1** 的整体思路推进。

已完成：
- Phase 0：数据准备与分层
- Phase 1：Reference Model 训练

现在进入：
- **Phase 2：半离线计算 token score / token mask**

根据现有实验设计，RHO-1 风格改造只作用于 **CosyVoice2 的 text-speech LM 阶段**，不修改 tokenizer，不在 CFM / vocoder 阶段引入该机制；工程实现采用 **方案 B：半离线打分**。Phase 2 的核心任务是：在正式训练集 `D_train` 上，利用 **当前 target model** 与 **冻结的 reference model** 对同一批样本做双模型前向，逐个 **speech token** 计算 loss，并进一步计算 excess loss，再生成 token-level score 或 mask，供后续 selective training 使用。

---

## 你现在要做的事

请只实现 **Phase 2** 所需的代码与必要改动。

目标如下：

1. 读取正式训练集 `D_train` 的 batch 或 shard
2. 对同一批样本 `(y, μ)`：
   - 用 **当前训练中的 target model** 计算每个 speech token 的 loss
   - 用 **冻结的 reference model** 计算同一位置 speech token 的 loss
3. 计算每个 speech token 的 **excess loss**：

```text
L_delta(μ_i) = L_target(μ_i) - L_ref(μ_i)
```

4. 基于 `L_delta` 生成：
   - token-level score
   - 或 top-k% token mask
5. 将结果保存到磁盘，供后续 Phase 3 使用

---

## 核心定义

对每个样本：

```text
(y, μ)
```

其中：
- `y`：text token 序列
- `μ`：speech token 序列

对每个 speech token `μ_i`，需要得到：

```text
L_target(μ_i)
L_ref(μ_i)
```

并计算：

```text
L_delta(μ_i) = L_target(μ_i) - L_ref(μ_i)
```

这就是本阶段最核心的 token score。

---

## 代码纪律（必须严格遵守）

这是最高优先级要求。

### 1. 不要添加本文档没有明确要求的功能

禁止私自加入下列内容：

- 不要提前实现 Phase 3 selective backward
- 不要修改 tokenizer 训练逻辑
- 不要修改 CFM / vocoder 训练逻辑
- 不要顺手重构整个训练框架
- 不要增加复杂配置系统
- 不要增加实验性 score function
- 不要增加复杂可视化、监控面板、数据库缓存等扩展模块
- 不要加入本文档未要求的“优化性功能”

### 2. Keep It Simple and Stupid

编程时请始终遵守：

- 优先直接、清晰、可验证的实现
- 优先复用现有 CosyVoice2 工程中的 dataset / batch / model forward
- 优先少改主干、多加独立脚本或小模块
- 避免过度抽象、过度封装、过度设计

### 3. 以现有工程为准，不发明新流程

如果工程细节不完整：
- 先基于当前代码结构做合理、保守的实现
- 不要自行创造全新的训练路径
- 不要因为“可能以后有用”而引入额外层次

---

## 本阶段的实现边界

Phase 2 只负责 **半离线 token scoring**，不负责正式 selective training。

本阶段应完成的范围：

1. 双模型前向
2. per-speech-token loss 提取
3. excess loss 计算
4. top-k 选择
5. score / mask 落盘

本阶段不应完成的范围：

1. selective backward
2. selective optimizer step
3. 训练主循环中的在线 mask 刷新
4. 其他阶段的实验逻辑

---

## 推荐实现方式

请优先采用对现有工程侵入较小的方式。

### A. 新增一个独立的 Phase 2 scoring 脚本

例如可使用类似路径：

```text
scripts/rho1_phase2_score.py
```

职责：
- 载入 `D_train`
- 载入 target checkpoint
- 载入 reference checkpoint
- 执行双模型前向
- 计算 per-token loss 与 `L_delta`
- 生成 token score / token mask
- 保存结果

### B. 新增一个简洁 helper 模块

例如：

```text
cosyvoice/rho1/scoring.py
```

职责尽量单一，只包含本阶段确实需要的函数，例如：
- 提取 speech token 对应位置
- 计算 per-token CE loss
- 计算 `L_delta`
- 构造 top-k mask

---

## 计算要求

### 1. 只关注 speech token loss

请只对 **speech token 部分** 计算或提取 loss：

- 不是全序列 token
- 不是 text token
- 只比较 speech token positions

### 2. target / reference 必须严格对齐

要求：
- 同一批输入
- 同一 speech token 位置
- 同一 CE 对齐方式
- 同一个有效 token 集合

### 3. reference model 的使用方式

要求：
- `reference model.eval()`
- 参数冻结
- 只参与前向
- 不做 backward

### 4. target model 的使用方式

在这个 scoring 脚本里也可以 `eval()`，因为本阶段只做打分，不做训练。

---

## 推荐函数粒度

建议实现一个清晰的函数，例如：

```python
compute_speech_token_losses(model, batch) -> Tensor
```

目标返回：
- speech token 对应的 per-token loss
- shape 可为 `[B, T_speech]`，或其他可明确映射到 speech token 的形式

然后执行：

```python
target_loss = compute_speech_token_losses(target_model, batch)
ref_loss = compute_speech_token_losses(reference_model, batch)
delta = target_loss - ref_loss
mask = build_topk_mask(delta, ratio=topk_ratio)
```

---

## top-k 选择要求

第一版只需要支持最直接的 top-k 策略：

- 在当前 batch 或当前 scoring shard 内
- 按 `L_delta` 从高到低排序
- 选择 top-k% token
- 生成布尔 mask

默认先支持：

```text
top_k_ratio = 0.6
```

可以暴露为命令行参数，但不要引入复杂参数系统。

---

## padding / ignore / 有效位置处理

请特别注意：

1. padding token 不能参与打分
2. ignore_index 对应位置不能参与打分
3. 非 speech token 位置不能参与打分
4. 不同长度样本下，有效位置必须清晰可控

最终 score / mask 只应覆盖 **有效 speech token**。

---

## 输出要求

请生成可供后续 Phase 3 读取的结果文件。

输出格式保持简单直接，例如每个 batch / shard 一个 `.pt`、`.npy` 或 `.pkl` 文件，内容可类似：

```python
{
    "sample_id": ...,
    "speech_token_score": ...,
    "speech_token_mask": ...,
    "topk_ratio": ...,
}
```

只保存后续训练真正需要的信息，不要额外堆砌冗余字段。

---

## 交付要求

完成后，请向我汇报以下内容：

### 1. 改了哪些文件
直接列出路径。

### 2. 每个文件的作用
每个文件一句话说明。

### 3. 运行命令
给出最直接的可运行命令，例如：

```bash
python scripts/rho1_phase2_score.py \
  --train_manifest ... \
  --target_ckpt ... \
  --ref_ckpt ... \
  --output_dir ... \
  --topk_ratio 0.6
```

### 4. 输出文件说明
说明每个输出文件里保存了什么。

### 5. 验证方法
告诉我如何验证以下几点：
- target / ref 的 per-token loss 是否都已正确算出
- `L_delta` 是否已成功生成
- mask 是否只覆盖 speech token
- top-k 比例是否正确

---

## 编码风格要求

请遵守以下风格：

- 代码简洁
- 变量命名清晰
- 注释克制，只写必要信息
- 可读性优先
- 工程一致性优先
- 不写与当前任务无关的框架代码

---

## 一句话目标

请基于当前 CosyVoice2 工程，完成 **Phase 2 的半离线 token scoring 模块**：

- 对 `(y, μ)` batch 做 target / frozen reference 双模型前向
- 提取每个 speech token 的 loss
- 计算 `L_delta = L_target - L_ref`
- 生成 top-k% token score / mask
- 保存到磁盘
- 严格限制实现范围，不添加本文档未要求的功能
