# Codex Prompt — Phase 1: 训练 Reference Model（RM / 影子模型）

## 0. 任务定位

现在开始实现 **Phase 1**。

这是本项目中第一次真正开始“训练”，但请注意：
**这一步训练的不是最终 target model，而是 reference model（RM / 影子模型）。**

它的作用不是最终生成语音，而是学习“高质量 reference corpus 的 token 分布”，并在后续 Phase 2 中为 target model 提供 **token-level loss baseline**。

---

## 1. 本阶段目标

请你只实现以下目标：

- 使用 **Phase 0 已经准备好的 `D_ref`**
- 在 **CosyVoice2 的 text-speech LM** 上训练一个 **reference model**
- 使用 **标准 cross-entropy / next-token prediction**
- **不做 token 选择**
- **不做 excess loss**
- **不做 selective training**
- 训练结束后导出一个 **可冻结使用的 RM checkpoint**

简而言之：

**Phase 1 = 用 `D_ref` 训练一个标准 reference LM。**

---

## 2. 你必须理解的核心事实

### 事实 1：这一步已经开始训练
是的，Phase 1 就需要真正开始训练了。

### 事实 2：训练对象仅限 LM
只训练 **CosyVoice2 的 text-speech LM**。  
不要把训练扩展到：
- tokenizer
- CFM
- vocoder
- 任何声学生成后端

### 事实 3：训练数据仅限 `D_ref`
Phase 1 只允许使用：
- 高质量 reference subset：`D_ref`

不要混入：
- `D_train`
- random subset
- target training shard
- 任何 Phase 2/3 才会用到的数据结构

### 事实 4：训练目标是普通 CE
就是标准的 speech-token 自回归预测：
- 根据输入序列预测下一个 token
- loss 为标准交叉熵
- 不做 token mask 筛选
- 不做 top-k
- 不做打分融合

---

## 3. 代码纪律（最高优先级）

## 第一原则
**不要添加任何主 md 文档没有提到的功能。**

## 第二原则
**Keep it simple and stupid.**

如果某个东西不是 Phase 1 明确需要的，就不要实现。

---

## 4. 这次只允许做什么

你这次只能做以下事情：

1. 读取 Phase 0 输出的 `D_ref` 数据
2. 构建 Phase 1 训练用 dataloader
3. 复用或接入现有 CosyVoice2 text-speech LM
4. 加载与 target model 一致或尽量一致的初始化方式
5. 用标准 CE 跑 reference model 训练
6. 保存 checkpoint
7. 记录最基础的训练日志与验证指标
8. 导出后续 Phase 2 可直接读取的冻结 RM

---

## 5. 明确禁止做什么

以下内容一律不要做，除非我后续明确要求：

### 不要越界到 Phase 2 / 3 / 4
不要实现：
- per-token excess loss
- `L_target - L_ref`
- token score
- token ranking
- top-k% token selection
- token mask 生成
- selective loss 回传
- 半离线打分逻辑
- target model 正式训练
- CFM 训练
- vocoder 训练

### 不要改模型定义
不要：
- 重写一套新的 LM 架构
- 擅自替换 backbone
- 改 tokenizer 训练方式
- 重训 tokenizer
- 改 CFM
- 改 vocoder

### 不要做复杂扩展
不要增加：
- Web UI
- 可视化面板
- 服务化接口
- 数据库
- 大型实验管理系统
- 复杂 callback 框架
- 多余的分布式封装
- 过度抽象的 Trainer 重构

### 不要“顺手优化”
不要因为“看起来以后可能用得上”就提前实现：
- target/RM 双模型联合框架
- online scoring
- 多任务 loss
- utterance prior 融合
- HRM/LRM 混合打分
- RL / DPO
- instruction tuning
- streaming/non-streaming 双模式新逻辑改造

---

## 6. 工程目标

请你把这一步做成一个 **最小可行且明确可复现的 RM 训练模块**。

它应该满足：

- 能单独读取 `D_ref`
- 能单独训练 RM
- 能输出可冻结 checkpoint
- 能和后续 Phase 2 解耦
- 不污染 target model 主训练逻辑
- 不改动不相关代码

---

## 7. 建议实现方式

### 7.1 输入
输入应来自 **Phase 0 的标准化产物**。  
优先复用 Phase 0 已经生成的统一样本表示，例如：

- `sample_id`
- `text`
- `y`（text tokens）
- `μ`（speech tokens）
- split / subset 标记
- 其他必要元信息

如果 Phase 0 已经提供 manifest，请直接读取，不要重新设计第二套数据格式。

---

### 7.2 数据子集
你需要确保训练集来源为：
- 仅 `D_ref`

如果需要验证集：
- 可以从 `D_ref` 内部切出一个很小的 val split
- 但不要引入新的复杂 split 体系
- 不要重新做一套数据筛选逻辑

---

### 7.3 模型
训练对象必须是：
- **CosyVoice2 的 text-speech LM**

原则：
- 结构与未来 target model 保持一致
- 初始化方式尽量一致
- 但不需要与 target model 联动实现

如果仓库里已有 CosyVoice2 LM 训练入口，优先复用。
不要重写一套新模型，除非现有代码完全不可用。

---

### 7.4 训练目标
标准 CE 即可。

概念上你要做的是：

- 构造 CosyVoice2 的 LM 输入序列
- 在 speech token 预测目标上做 next-token prediction
- 按现有 CosyVoice2 的 ignore mask / loss mask 规范处理
- 只保留标准训练逻辑

请记住：
**这里不要引入任何 RHO-1 的“筛 token”机制。**
Phase 1 的 RHO-1 对应关系，只体现在：
- “高质量 reference corpus”
- “训练一个 reference model”
而不是 selective training 本身。

---

## 8. 你应该优先复用的东西

如果仓库中已经有以下能力，请尽量复用而不是重写：

- 现成 dataset / collator
- 现成 LM forward
- 现成 loss mask 构造
- 现成 trainer / 训练脚本
- 现成 checkpoint 保存逻辑
- 现成配置模板

你的原则是：
**最小侵入，最小修改，最小新增。**

---

## 9. 你应该新增/修改的内容

建议你将 Phase 1 实现限制在少量文件中，例如：

### 可以接受的新增内容
- 一个专门的 Phase 1 训练脚本
- 一个简洁的 Phase 1 配置文件
- 一个 `D_ref` 专用 dataset loader 或 wrapper
- 一个 RM checkpoint 导出脚本（如有必要）

### 可以接受的少量修改
- 在现有训练入口中增加一个 `mode=reference_lm` 分支
- 在现有配置系统中增加 `phase1_reference` 配置
- 在现有 dataset 读取逻辑中支持仅加载 `D_ref`

### 不应做的修改
- 大规模重构项目目录
- 改掉主训练逻辑
- 修改 tokenizer/CFM/vocoder 模块
- 为未来 Phase 2/3 提前埋很多复杂接口

---

## 10. 配置要求

请配置尽量简单，并保证这些信息清晰可控：

- 数据路径（Phase 0 输出）
- `D_ref` manifest 路径
- batch size
- learning rate
- max steps 或 epochs
- warmup（如已有现成配置）
- save interval
- eval interval
- output dir
- random seed

不要做复杂配置继承系统。  
不要为了“灵活性”引入过多层级。

---

## 11. 日志与监控要求

只做最基本且必要的日志：

- train loss
- eval loss（如果有 val）
- global step
- 学习率
- checkpoint 保存信息
- 失败样本 / 数据问题计数（如果容易复用）

不要接入复杂日志系统。  
不要加花哨图表。  
不要写庞大的 callback framework。

---

## 12. checkpoint 要求

训练完成后，至少要能输出：

### 必需产物
- 一个可加载的 RM checkpoint
- 一个最基础的训练配置快照
- 一个最基础的训练日志或 summary

### 可选产物
- best checkpoint
- last checkpoint

但不要把 checkpoint 管理写得很复杂。

---

## 13. 成功标准

只有满足以下条件，这次实现才算成功：

1. 能读取 `D_ref`
2. 能正常跑通 reference LM 训练
3. 训练对象确实是 CosyVoice2 的 text-speech LM
4. 使用的是标准 CE，而不是 selective loss
5. 能导出可冻结的 RM checkpoint
6. 代码实现范围没有越界到 Phase 2+
7. 没有新增主文档未要求的功能

---

## 14. 希望你交付的内容

请直接给我以下内容：

1. 你准备新增/修改的文件列表
2. 每个文件的职责
3. 关键代码实现
4. 训练命令
5. 配置示例
6. checkpoint 输出位置
7. 最小运行说明
8. 你做了什么
9. 你明确没有做什么

最后请单独输出一个：

## Scope Check

逐条确认：

- [x] 这一步确实开始训练了
- [x] 训练对象仅为 reference model
- [x] 只使用 `D_ref`
- [x] 只训练 CosyVoice2 text-speech LM
- [x] 使用标准 CE
- [x] 没有实现 excess loss
- [x] 没有实现 token scoring
- [x] 没有实现 selective training
- [x] 没有进入 Phase 2 / 3 / 4
- [x] 没有改 tokenizer / CFM / vocoder
- [x] 没有添加文档未要求的新功能
- [x] 保持 KISS

---

## 15. 最后的强提醒

请一直记住：

### 这一步是“训练 reference model”，不是“训练整个新系统”。

### 这一步的本质是：
**在高质量子集 `D_ref` 上，训练一个标准的 CosyVoice2 text-speech LM，产出后续可冻结使用的 RM checkpoint。**

除了这件事之外，其他都先不要做。

### 再强调一次：
- 不要添加任何主 md 文档没提到的功能
- Keep it simple and stupid
- 能复用现有代码就复用
- 能少改就少改
- 不要提前实现 Phase 2