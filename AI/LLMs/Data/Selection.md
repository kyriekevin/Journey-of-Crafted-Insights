---
tags:
  - llm
  - data
  - sft
  - ct
  - pretrain
date: 2025-01-06T11:06:00
---

# SFT Data Selection

## Why?

1. 随着业务发展，不同的任务数据不断积累，子任务数据过大可能会影响其他任务的泛化能力。因此，需要有方法来精简数据。
2. 对于业务的一个任务场景，没有这个场景的专用SFT训练数据，需要有方法从开源数据中筛选出对这个场景有增益的数据。
3. 探索训练集中哪些数据对具体业务场景带来增益最大。

## Methodology

1. Non-Target Method：针对从大训练数据集中进行筛选，没有单一优化的任务场景
2. Target Method：针对具体任务场景，从数据集中挑选出对这个场景有增益的数据

## MoDS

paper: [MoDS: Model-oriented Data Selection for Instruction Tuning](https://arxiv.org/pdf/2311.15653)
github: [MoDS](https://github.com/CASIA-LM/MoDS)
Methodology: Non-Target Method

### TL;DR

围绕三个点：quality、coverage和necessity

1. 使用reward-model-deberta-v3-large-v2从候选数据中筛选出高质量数据
2. 使用k-center greedy聚类算法，从第一步筛选出的数据中挑选多样性数据，即Seed Instruction Data
3. 用第二步筛选出的Seed Instruction Data训练，训练后对第一步得到的高质量数据集算loss，loss高的就是necessity数据
4. 混合High-Quality Instruction Data和Augmented Instruction Data，进一步训练模型

### Highligh

1. 考虑比较全面，考虑了SFT数据的质量、多样性和必要性

### Lowlight

1. 依赖外部评分模型（有偏），很难确保筛选出的数据是真正的高质量数据
2. 流程相对复杂，涉及多个步骤算法

## CaR

EMNLP 2024
paper: [Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation](https://arxiv.org/pdf/2402.18191)
github: [CaR](https://github.com/IronBeliever/CaR)
Methodology: Non-Target Method

### TL;DR

和MoDS类似，融合了质量和多样性的要素

1. 使用评分模型对数据进行质量评估，筛选出高质量数据作为训练集的一部分
2. 采样k-means聚类，从每个簇中选取高质量数据，作为训练数据的另外一部分

论文讨论了怎么得到一个更符合人类喜好的评分模型，很多评分模型是倾向于GPT偏好的，而不是人类喜好的。

### Highligh

1. 兼顾数据质量和多样性
2. 操作流程相对于MoDS简单

### Lowlight

1. 依赖外部评分模型（虽然给出了建立评分模型的经验和skills，但还是有偏）
2. 相对于MoDS，没有考虑数据的必要性

## Nuggets

paper: [One Shot Learning as Instruction Data Prospector for Large Language Models](https://arxiv.org/pdf/2312.10302)
github: [nuggets](https://github.com/pldlgb/nuggets)
Methodology: Target Method

### TL;DR

将每一条数据逐一作为one-shot example，观察加入后对每条测试数据输出的影响。如果加入某条数据能降低测试数据loss，则视为有增益效果。

### Highligh

1. 方法直观
2. 实现难度低

### Lowlight

1. 计算复杂度高
2. 测试集的质量对结果影响较大

## LESS

ICML 2024
paper: [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/pdf/2402.04333)
github: [LESS](https://github.com/princeton-nlp/LESS)
Methodology: Target Method

### TL;DR

LESS考察训练集给模型梯度优化方向对测试集loss的下降，选择对测试集loss下降最大的训练集作为gold train dataset

### Highligh

1. 方法简单
2. 对训练集进行统计，可以迁移到不同测试集上

### Lowlight

1. 计算复杂度高
2. 测试集的质量对结果影响大

## IFD

NAACL 2024
paper: [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/pdf/2308.12032)
github: [Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM)
Methodology: Non-Target Method

### TL;DR

1. 提出了不使用任何模型，只依靠大模型本身来自动挑选最适合该模型的高质量数据集方法。
2. 提出了指令跟随难度(Instruction-Following Difficulty, IFD) 这一度量标准，通过这个度量标准来评估数据集的质量。

### Abstract

如何在数据的质量和数量之间取得平衡一直是一个关键问题。为了优化模型，通常需要大量高质量指令数据，但手动整理筛选数据成本高。
现有的研究表明，数据的质量比数量更重要。因此，这篇论文提出了一种创新方法，通过模型自引导(Self-Guided)的方式，自动从开源数据集中选择最有价值的训练样本(Cherry Data)，5%-10%，以提高指令调优的效率和效果。

### Methodology

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061141269.png)

#### Learning from Brief Experience

使用k-means方法从原始数据集中选取少量数据（1k条）然后训练1个epoch，使得模型获得基础的指令跟随能力

k-means: 保证这个部分数据的多样性
1k&1epoch: 尽可能减少资源消耗 follow LIMA的setting

#### Evaluating Based on Experience

引入IFD指标(Instruction-Following Difficulty)，来判断LLM是否需要学习这个指令数据
- LLM在给定指令后，生成的相应回答的loss/perplxity
- LLM在不给定指令的情况下，生成相应回答的loss/perplxity

前者被命名为条件回答分数(Conditioned Answer Score, CAS)，后者被命名为直接答案分数(Direct Answer Score, DAS)

##### Conditioned Answer Score (CAS)

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061231071.png)

这个分数本质是LLM指令微调的目标函数，表征当前LLM在给定指令时，生成相应回答的难易程度。
这个分数不能直接用来评估指令本身的难易程度，因为受到预训练影响，LLM生成回答本身的难易程度也会影响这个分数。

##### Direct Answer Score (DAS)

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061358132.png)

这个分数表示在不给定instruction的情况下，模型直接生成回答的难易程度，本质是LLM的预训练目标函数。
- DAS低：LLM经过预训练后，对这个回答的句子比较熟悉，很容易生成
- DAS高：LLM经过预训练后，对这个回答的句子不熟悉，生成困难

##### Instruction-Following Difficulty (IFD)

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061402075.png)

1. 理论上IFD应该小于1，在给定指令之后，由于context更多，因此生成对应回答的loss/ppl应该更低。所以如果IFD值大于1，则说明这条指令数据的instruction和response可能不够align，需要舍弃。
2. IFD小于1，且值比较高，即CAS和DAS比较接近，说明instruction对response的生成影响不大，说明LLM还没有学会他们之间的alignment，说明指令对LLM来说比较困难有必要学习。
3. IFD小于1，且值比较低，说明在给定instruction之后，LLM很容易生成需要的response，说明这个指令对LLM来说比较简单，切没必要学习。

#### Retraining from Self-Guided Experience

挑选出IFD小于1且值比较高的指令数据

### Experiments

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061414398.png)

在Alpaca和WizardLM上进行实验，大约5%-10%的数据即可超过原来的模型

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061414648.png)

Low IFD score的performance差，进一步说明方法有效性，即高IFD的sample更有效，低IFD的sample反而对训练有负面影响

### Limitation

需要训练一个Instruct model, Instruct model的性能会影响到IFD的计算

### Lowlight

1. 高IFD样本未必真正代表困难样本：答非所问的错误样本或者偏短的样本，prompt和response的不相关性，自然导致IFD分数偏高。
2. IFD选择样本会容易挑选出某一特定任务的样本，忽略了数据的多样性

## Superfiltering

ACL 2024
paper: [Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/pdf/2402.00530)
github: [Superfiltering](https://github.com/tianyi-lab/Superfiltering)
Methodology: Non-Target Method

### TL;DR

1. 提出了Superfiltering的概念，揭示了在感知和评估指令微调数据难度方面，弱语言模型和强语言模型之间的强一致性。
2. 提出了利用SLM例如（GPT2）成功进行指令数据筛选的方法，大幅度减少数据过滤的时间和成本。
3. 提出的方法是即插即用的，不需要任何训练，不需要任何LLM参与，也不需要划分训练测试集，纯粹依靠语言模型本身的特性完成对数据的评估，得到的分数可以用来数据筛选，也可以作为对训练数据的整体直观评估。

### Abstract

现在数据筛选虽然可以自动化，但是这个过程通常需要使用强大的LLM参与，以确保所选择的数据具有足够质量。这种方法的主要缺点是筛选本身非常耗时，特别是在数据集非常大的情况下，进筛选过程就可能消耗大量计算资源和时间。
论文提出Superfiltering方法，通过使用SLM（GPT2）来替代LLM进行数据筛选，显著降低数据筛选的时间和成本。
尽管GPT2这样的SLM在规模和性能上远远落后于LLM，但在感知指令难度和进行数据选择方面和LLM表现出比较高的一致性。

### Methodology

论文同样使用IFD作为核心度量方法，但是不同在于：论文发现在去除模型本身输入的template的情况下，即使使用没有经过指令微调的base model，依然可以很好地挑选出合适的高质量数据。

为了验证不同强弱模型对指令困惑程度的感知，选取GPT-2系列SLM和Llama 2系列LLM，在多个数据集上对比困惑度和IFD score。

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061444920.png)

对于困惑度而言，不同强弱模型之间的scale有明显差距，越强的模型在同样的数据上困惑度越低。
对于IFD分数而言，不同强弱模型之间的scale基本一致，同时它们的分布（小提琴形状）也极其相似。

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061459145.png)

Rank Correlation指Spearman's $\rho$，表征两个排序之间的相似度。以Llama2-7B计算得到的困惑度和IFD分数作为参考，可以发现即使最弱的GPT2，在3个数据集上困惑度以及IFD分数有着较高的一致性。

困惑度：不同强弱模型困惑度scale相差会很大，但排序是有比较高一致性的
IFD：不论scale，分布还是排序，强弱模型都有较高的一致性

#### Superfiltering as Dataset Assessment

可以快速获得对数据集中每一条数据难易度的评估，不仅可以用来做数据筛选，也可以提供一种思路对数据及整体评估

Alpaca数据集：质量相对一般，有着较多低质量数据，因此小提琴图的上下部分较宽，表明数据质量存在显著差异。
WizardLM数据集：有ChatGPT 3.5生成，指令整体较复杂，但是依然存在一些质量较低的噪声数据，因此小提琴图中体现为顶部较宽且有一条长而细的尾巴。

#### Superfiltering with Diversity

假设在高质量数据子集上实施多样性指标比在质量参次不齐的整个数据集上效果更好。

1. 使用Superfiltering选择一个相对高质量子集
2. 利用Facility Location函数来压缩所选取数据

Facility Location函数能够在捕获多样性和确保数据中不同簇或区域的代表性之间取得平衡。

### Experiments

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061507484.png)

在不同测试集上使用Superfiltering筛选出来数据和全部数据训练模型对比

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061508824.png)

不同数据筛选策略，以及使用不同的模型做筛选得到的模型对比

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061508455.png)

不同的数据筛选策略，做筛选得到的模型的结果以及时间对比

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202501061526115.png)

为了进一步保证效率，使用"sentence-transformers/all-MiniLM-L6-v2" 作为编码器。
首先通过Superfiltering选择20%的数据，然后利用Facility Location函数进一步选择2%的数据
用2%数据训练的模型在性能上与使用全量数据训练的模型相当或更优

### Lowlight

1. 效果不如IFD方法

## Echo SFT Data Selection (WIP)

Methodology: Non-Target Method

instruction: quality, coverage, necessity

### Single turn

1. tagging: qwen + few shot -> channel loss + diversity
2. IFD: gpt + qwen2.5 SLM < 7B -> sampled data + lora + merge

### Multi turn

1. dedup & filter: rule + emb sim
2. tagging: qwen + few shot -> channel loss + diversity
3. reward model
