---
tags:
  - AI
  - llm
  - Continue-Pretrain
  - CT
date: 2024-11-11T10:12:00
---
# 背景
领域模型是在某一领域性能特别好的模型，可能是法律、医学等。一般来说，领域模型比较重要的场景是RAG，需要一个精确度高的检索工具和完善的知识库，来辅助模型回答问题。

那是否意味着，领域模型只需要做好sft和rlhf就可以呢？
其实在大部分任务场景下，80%模型的通用能力是可以cover住的，基本只需要一个开源模型+几千或者几万条sft语料就可以。但是，领域模型更多需要的是那20%的case，这些case需要模型具有领域思维、领域知识、领域语言风格等，这是sft所做不到的。

SFT和Pretrain主要区别就是数据量和多样性，Pretrain的数据量更大，多样性更丰富，和现实世界自然语料的分布更接近，因此不需要case by case的关注数据质量，更多的只要保证数据源的质量和可信度，多样性可以由其他数据源混合来提高。SFT可以认为数据是人类偏好、任务导向的数据，相比于自然世界的语料是有偏的。需要严格确保数据质量和丰富性，防止出现hacking。

continue-pretrain的目的便是让模型尽可能学习这个领域专业名词、专业知识，最终形成领域思维。但是，大量paper证明：continue-pretrain的过程，是学习新知识遗忘旧知识的过程。这也就表明在提高领域知识的同时，模型也在丢失通用能力。正如前面提到的，80%场景是需要模型通用能力来cover的。

因此我们continue-pretrain的目标是：在continue-pretrain阶段学习领域知识的同时，尽可能减少模型通用能力的丢失。

# 数据
continue-pretrain和pretrain在数据方面类似，详细参考[[Pretrain]]数据部分，整体Pipeline如下

### 数据收集
* 通用数据
	同样可以基于开源预训练数据集，并且考虑到base model不是自己训的，在训练集选择时间上一般可以选择开源模型发布前六个月时间节点之后的数据。
* 领域数据
	获取领域相关书籍、文档等，同样的问题是需要pdf解析
	对于领域数据，缺乏高质量pdf解析服务，只使用python解析，这部分有些脏的语料可以有以下三种使用方式：
	1. 作为pretrain一阶段训练语料（一阶段一般容忍有部分质量不够高数据）
	2. 作为合成数据语料，也就是下面会提到的知识密度，将这部分语料给到gpt做合成（gpt是可以处理这种有一定识别错误的语料，但合成后的数据质量依赖gpt原始在这个领域的水平）
	3. 作为sft合成数据语料，即让gpt根据这段有一定价值的语料提出qa对，具体参考[[SFT]]数据部分

> gpt提出的问题取决于gpt在该领域原始能力，经常提出问题过于通用，缺乏领域视角
>
> 通用数据和领域数据都是可以往数据合成方向走。合成数据其实是模型蒸馏的一个变体，合成数据是更大模型的输出数据作为Teacher，小模型作为Student从中学习知识。用A大模型合成数据对A大模型本身没有提高，一般使用同源模型中大的模型来合成。

## 数据清洗
* 通用数据
	和Pretrain流程一样，清洗基于打分器、规则和脱敏
* 领域数据
	领域数据更侧重于规则和脱敏

* 规则
	与通用数据不同，领域数据一般出现问题是特定短语话术特别高频，并且存在较多重复引用，领域名词多
	相较于通用数据，更多需要考虑链接、短语话术等清洗

* 脱敏
	领域数据可能涉及大量用户信息等敏感数据
	1. 对于包含用户信息数据直接丢弃？
	2. 对于涉及信息进行替换脱敏？

	直接丢弃可能会把大量有用数据丢弃，并且导致上下文不连贯
	简单替换脱敏可能会导致脱敏的这个pattern大量重复出现，导致模型能力下降
目前对于对话数据，是采用每一条都当做一个text，对于敏感语句直接丢弃。潜在问题可能是每个text信息密度低，且缺失上下文。

## 数据去重
去重方法：都是基于MinHash这类hash算法居多
去重粒度：document和sentence粒度居多，领域可能都可以考虑基于字符

- 领域数据
	- 领域数据是有必要精细化的清洗，尤其是特定的短语 pattern应该尽可能避免，这些pattern会导致loss降低明显，甚至影响模型性能。客服领域在测case中发现这些小pattern占比高，模型有明显过拟合现象。
	- 在商汤做function call发现，虽然希望模型输出json格式，但模型只要不到1k条，就可以掌握强json格式，通用能力却显著降低。即使使用json markdown等多个格式数据试图缓解通用能力，还是会有问题。
* 通用数据
	* 采用一个开源预训练数据集，在这个基础上做增量式扩充数据集，可能是一个更实际的方式。
	* 并且尽可能选取新的开源数据，一个比较好的判断方式是选取基模发布半年前时间节点之后的数据。考虑的基座团队往往从数据收集、清洗到进行pretrain、sft、rlhf基本是半年的时间跨度。后出的数据往往是之前数据上做更严格清洗或者合成强化得到的更高质量数据。

## 数据配比
和Pretrain不同，continue-pretrain涉及两个配比：
1. 通用数据内部配比
2. 通用数据和领域数据配比

* 通用数据内部配比
	和pretrain不同地方在于，continue-pretrain可能需要更多考虑80%的通用能力是否有所侧重，根据具体下游任务case来调整不同task之间的配比
	同时语言的配比，可能也是需要调大中文的占比，而不再是中:英:Code=4:4:2
* 通用数据和领域数据配比
	一般而言通用数据还是占大头，一般是1:4到1:9之间

> llama3和面壁智能给出的数据配比，得出的结论：Code重要，英文也很重要。
> 部分paper观点是：general knowledge基本来自英文语料，中文语料更多是起到对齐和迁移作用

continue-pretrain可以分成三大类：领域知识、语言类和long context
这几类在训练和学习上有不小差距：
* 领域知识：
	llm中或多或少有一些这样知识，初始loss一般比较低，遗忘程度也低，最优配比低
* 语言类和long context：
	语言类是因为语言上gap，初始loss一般比较高，随着训练loss会稳定下降，但遗忘程度高，最优配比高
	long context主要是对资源消耗高，遗忘程度高，最优配比高

## 数据顺序
* 通用数据
	和Pretrain一致，考虑课程学习和In context pretrain，同样应该准备不同Stage数据
* 领域数据
	领域数据是否也可以进行课程学习和In context pretrain呢？
	1. 完整用户和客服的对话（是否去合成？）
	2. Stage one知识库+书籍+对话，Stage two知识库+书籍+高质量的下游任务？

> 更进一步，像llama系列的In context pretrain的话，是不是可以把相关会话拼在一起？但涉及之前清洗逻辑，整个text应该是语义连贯的。
>
> 以及客服回复中有很多工具使用+锦囊话术，这些模型是无法输出的，学习了就会变成幻觉（我查看了xxx，之前进线的工单，是否可以透传等）。更合适做法是否是ReAct+RAG？


## 数据总结
* 数据处理
	1. 训练bert系列进行数据分类、打标，然后进行清洗去重
	2. 使用同源大模型做reward model，然后进行清洗去重 具体参考[[同源模型]]
	3. 使用一个大开源预训练数据集，以这个数据集为base做增量，新增数据集按base的分类重新分类，并清洗去重（开销最小，短期拿收益）
* 数据质量
	数据质量上，continue-pretrain通用数据不用那么精细，目标是通用能力不下降而不是，通用能力大幅提升。qwen2技术报告里提出，训练12T数据的模型和训练7T数据的模型，模型能力上基本没有提升（额外的5T数据质量不如7T，卡的阈值小），没有带来收益。
	而我们是拿不到qwen或者llama系列pretrain那样高质量数据，所以再如何细致化清洗通用数据，收益可能都不大。
	也许更合适做法是基于一份开源预训练数据集，做增量收集处理。
* 数据分块
	在数据处理时就提前分块分Stage，配合训练提到的Loss分析
	1. 分Stage：Stage one和Stage two，Stage two在Stage one基础上补充更高质量数据以及sft数据
	2. 分块：以B为单位做增量式训练，避免中间训崩，且可以较好观察loss以及对比质量


# 训练
## 模型选型
### Model Zoo
1. 选同源模型size多的 [[同源模型]]
2. 选模型能力强的，参考 [opencompass](https://rank.opencompass.org.cn/home)
3. pretrain训练本身就支持中文的（避免从llama系列ct，需要扩词表+2次ct）

### Model Size
选择模型的size一般不是根据场景需求决定（除非是Math Reason等负责任务，必须使用大模型才能保证效果），小模型能力上限是未知数。
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411111419589.png)
可以看到Qwen 2.5绘制的这个MMLU能力曲线图，随着模型发布时间的推移，越来越小的模型可以达到之前较大模型的水平。基本不会出现，模型size选择小了，后续业务场景满足不了的情况，基本可以通过sft阶段救回来，无非就是让模型在业务能力上过拟合。

* 训练算力：给ct分配的机子数量，训练多久ct，训练多少B数据，以及使用的训练框架每天可以训练多少B token。这些都是在正式提交任务就可以估算确定的。
* 推理算力：AutoModelForCausalLM 单卡A100 80G基本模型size极限就是30B+，并且一直是打满显存；双卡A100 80G基本模型size极限就是70B+，稍微超过一些seq len就OOM（不考虑量化的情况）。也就是说，选择模型size时候需要根据下游任务部署的推理机器决定，不要出现单卡刚好装不下模型的情况。时延等问题可以通过推理框架来解决。

## 训练框架
CodeBase：LLama-Factory
megatron vs deepspeed （一般T级别token训练量必须megatron，B级别两者都可以）
LLama-Factory 目前支持deepspeed [issue](https://github.com/hiyouga/LLaMA-Factory/issues/2956)，当后续ct数据量持续扩大，deepspeed的算力损失和debug上的问题会更加明显
并且LLama-Factory也不支持channel loss，单一的loss其实用wandb展示出来没什么大意义（可能需要大量沙盒实验，才能清楚loss的规律） [issue](https://github.com/hiyouga/LLaMA-Factory/discussions/5137)

## Stage
和Pretrain一样，ct一般也是two stage训练。Stage two数据质量需要远高于Stage one！

Stage one
warmup：在训练过程中，将学习率慢慢提高
constant / linear / cosine decay：维持稳定学习率，或者缓慢衰减学习率

Stage two
Anneal：用小学习率学高质量数据，IFT数据，Math Code Reason数据，提升通用模型逻辑能力（刷榜必备）
LLama paper和面壁智能技术报告都有提到这个trick
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411111459830.png)

## 训练超参
warmup: 当模型经过充分训练后，不管多长的warmup步数最后性能都差不多。一般设置为epoch * 1%就可以
lr：size越大的模型，模型表征能力和学习能力越强，lr通常需要设置小一些。
	小学习率：domain学得慢，general遗忘慢
	大学习率：domain学的快，但lr大到一定程度时，domain的loss下降速度会出现震荡。因为知识的学习速度是有上限的，且潜在数据分布差异导致模型不能用太快速度学习。
	不同size学习率是不一样的，基本是5e-6到5e-4一个区间内
优化器：模型优化是由动量+当前数据梯度所决定的，所以即使遇到脏数据，优化器也可以最大程度限制更新幅度。模型是具备一定抗噪能力。这也是pretrain和continue-pretrain需要warmup的原因，需要让模型积累一定动量来抵抗噪声。
bs：一般而言对于pretrain和continue-pretrain，4M bs比较合适。且较大的batchsize可能可以达到更低的Loss。

## 训练Loss
1. channel loss：对于continue-pretrain，一般channel loss会设置这么几个channel（两大类，和Pretrain类似）
	1. 按照语言分：那么就是zh、en、code和domain四个channel loss（domain数据不归入zh和en）
	2. 按照task分：那么就是math、code、reason等general task，以及domain task
2. loss spike：loss突然增加或者突然减小（除了epoch>1又见过一遍的情况），大概率是数据有问题（脏数据不仅可能导致loss高，也可能导致loss低）。出现loss spike就从上一个ckpt resume

有了channel loss应该如何分析呢？
理想的channel loss：
1. general loss：不管按语言还是按task分，loss基本持平，缓慢下降
2. domain loss：loss下降明显

并且channel loss是可以用来反推pretrain的数据配比和数据质量的
1. 初始loss低：任务简单，或者模型已经训过这个数据。loss低也可能是脏数据，全部是重复token或者固定pattern明显。
2. 初始loss高：模型没有见过这个数据，是新的要学的数据。（但是仍可能是脏数据，比如全是错别字等）
3. loss持平或缓慢下降：大概率是和pretrain数据配比相近
4. loss快速下降：数据容易学，可能是domain数据具有明显特点或者pattern，通用数据可能是强格式化数据（最好不用，强格式数据模型能力下降明显）
5. general loss下降明显：general数据对模型来说不够general，更像是新的domain数据，说明和原始pretrain数据配比偏离明显
6. domain loss下降明显：说明模型学到了新domain知识
7. domain loss不下降：初始loss低大概率是模型学过domain数据；初始loss高但loss不下降，数据不够干净或者数据比较难学
8. loss异常：loss上升、loss到0等等异常，大概率是lr设置问题以及代码环境问题

# 评测

## PPL
其实就是看测试集的loss来衡量模型效果，同样应该是channel loss。应该在domain上ppl有明显下降，但是general上ppl基本维持。

> ppl只能是和自己模型比，因为不同模型tokenizer不同，loss也没有可比性。但正常loss在pretrain会低于2，continue-pretrain后应该是在1-2之间（A卡loss）

## Benchmark
考虑到已经进行过continue-pretrain（假设有退火阶段），那么开源通用benchmark是有一定可信度的。但因为需要和原模型对比，尽量还是改造（pretrain团队基本都刷过这些benchmark，之前有一个工作专门测试测试集泄露比，国内开源模型基本都有问题。常见刷榜方式是在退火阶段直接将benchmark测试集进行训练）

所以如果在benchmark上掉点了，可能只是缺少退火阶段，或者退火阶段少混sft数据了

如果需要改造benchmark，如何改造？
1. 更改形式：不使用ppl计算ABCD概率，变成RAG形式Q A_A,Q A_B,Q A_C Q A_D，然后让模型回答Q
2. 更改选项：将ABCD换成1234、一二三四等，将正确答案改成其他选项都错误等
通过改benchmark真实测试模型能力

### 评价指标
* 客观评价：选Acc为衡量指标的数据集（能是多选最好是多选），bleu、rouge等指标基本都被淘汰了
* 主观评价：pk > score，score随机性特别大
	pk需要注意：
	1. AB答案顺序，gpt4有顺序偏好
	2. AB答案长短，gpt4有长度偏好
	3. AB答案越接近，随机性越大
	因此pk看不输的概率，同时pk最好方式还是让模型根据不同维度都给出pk结果

更重要可能还是实际体验，往往经过rlhf后的模型实际体验会更好，但是在榜单上表现会下降。

# Reference
[D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models](https://arxiv.org/pdf/2406.01375)
[Continual Pre-Training of Large Language Models: How to (re)warm your model?](https://arxiv.org/pdf/2308.04014)
[IN-CONTEXT PRETRAINING: LANGUAGE MODELING BEYOND DOCUMENT BOUNDARIES](https://arxiv.org/pdf/2310.10638)
[MiniCPM](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)
[QWEN2 TECHNICAL REPORT](https://arxiv.org/pdf/2407.10671)
[The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
[Qwen2.5: 基础模型大派对！](https://qwenlm.github.io/zh/blog/qwen2.5/)
