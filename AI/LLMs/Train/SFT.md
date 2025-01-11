---
tags:
  - llm
  - sft
date: 2024-11-12T18:39:00
---
# 背景
需要对模型进行SFT主要原因是，当zero shot模型输出不了理想推理结果时，使用one shot或者few shots来调整模型，可能也不是总有效的，尤其是在size小的模型上。
通过在prompt中增加shots或者cot来优化模型推理，这样的策略是有局限性的。
1. prompt中的shots和cot是会占用上下文窗口的空间，导致用于包含其他有用信息的窗口空间减小。
2. prompt中增加shots和cot会导致首token时延明显增大，并且后续每一轮时延也会增大。
3. 最重要的一点是，即使加了这些引导prompt，在实际应用中也不是完全有效。

和Pretrain以及Continue-Pretrain不同，sft是一个监督学习过程，可以使用标注数据来更新模型权重，让模型更好完成下游特定任务。sft的价值在于：优化模型输出体验，最大化利用文本窗口大小，提升模型实际应用效果。

进行领域任务sft，同样有以下几种训练模式（以及一些变体），会根据领域任务、训练样本数量和业务需求来选择合适的训练模式。
1. 基于base/chat模型+领域sft
2. 基于base/chat模型+领域ct+领域sft（只考虑领域任务效果的话）
3. 基于base/chat模型+领域ct+领域通用sft
4. 基于base/chat模型+领域通用ct+领域通用sft（目前采取形式）
5. 基于base/chat模型+领域通用ct（混入sft等高质量数据）+领域通用sft

有两个需要考虑的问题：
1. 是否需要Continue-Pretrain？
	大模型的知识来自于Pretrain阶段，考虑到客服领域和通用Pretrain数据分布差异比较大，同样Pretrain训练样本基本覆盖不到抖音客服知识，甚至基座基本注入不少是自家公司的知识和字节抖音知识可能是冲突的。并且如果我们领域数据量大（token > 1B）基本是需要进行Continue-Pretrain。
2. 是选择Chat模型还是Base模型？
	一般来说如果ct数据质量比较高且数量足够的话，且有足够多高质量的sft数据让模型对话等能力提升，那么从Base模型开始continue-pretrain更合适。

# 数据

## 组成形式

sft和pretrain在训练方式没有区别，主要区别就是在数据组成形式上
1. pretrain数据会padding成cutoff的长度；sft数据每一条是多长就是多长
2. sft会引入pretrain阶段没见过的special token，用于学习新的语义
3. sft会让模型见到eos_token，pretrain模型没有见过eos token无法停止生成（LLama-Factory中pretrain是没有加eos token）
4. 通过special token，sft语料分为不同角色部分（system，human，gpt）。并且因为sft时system和human的prompt往往比较同质化，会进行loss mask。那么对于session数据，就需要考虑是每一个answer都进行计算，还是只对最后一轮answer进行计算。
pretrain主要还是在学习知识，sft则是在做题，能在下游任务上有一个更好的指令follow能力。但这也意味着不应该在sft强行给模型注入知识，否则会使得模型通用能力掉点严重。
