---
tags:
  - llm
  - AI
  - Agent
date: 2024-11-04T10:16:00
---

# Agent 是什么？

Agent 是一个经典概念，强化学习入门时候就多次提及 agent，代指环境交互的实体。 学界对 Agent 有更 high-level 的期待，例如维基百科对 Intelligent Agent的定义：

> In intelligence and artificial intelligence, an intelligent agent (IA) is an agent that perceives its environment, takes actions autonomously in order to achieve goals, and may improve its performance with learning or acquiring knowledge. An intelligent agent may be simple or complex: A thermostat or other control system is considered an example of an intelligent agent, as is a human being, as is any system that meets the definition, such as a firm, a state or a biome.

Agent 核心能力是完成任务（achieve goals）、获取知识（acquiring knowledge）和进化（improve）

## LLM Agents 是什么？

基于Prompt与大模型交互的方式更像是静态的“输入-输出”，LLM Agent给大模型提供了一个动态决策框架，让模型有能力处理更复杂、更多样化的决策任务，以“语言”的方式参与“真实世界”的行动交互。

在此提供三个视角由浅入深供大家参考：
1. 产品定义
2. 直觉定义（直觉上如何复现这样的产品）
3. 学术定义

### 产品定义

定义：**基于大语言模型并能帮助某类用户的产品，就是LLM Agents**
例子：
1. 外语口语软件使用Chatgpt来帮助用户提升口语
2. 对于开放世界游戏玩家，游戏中NPC能够通过自然语言的方式同玩家进行交互（交互可以是游戏玩法或者是提供游戏情绪价值）

### 直觉定义

定义：**只要是以LLM为核心的产品，这个产品在一定外部环境下交互，有一定的内部模块，能够进行某种形式的输入以及进行某种形式的输出，就是LLM Agents**

这是一种偏直觉方式来认识LLM Agents，强调了LLM Agents的基本性质和特点，提炼相关要素：
1. 以LLM为核心
2. 外部环境
3. 内部模块
4. 输入和输出

例子：Perplexity 这样可以联网搜索的LLM工具。对应上面要素分别为：和用户交流以及和互联网搜索的外部环境，进行搜索并处理结果的相关模块，接收用户输入并生成回复输出

### 学术定义

参考复旦NLP组综述提出LLM Agents框架
![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202411242023901.png)


Agent是“运动”在“环境”（Environment）中，环境描述了Agent所在的状态空间，运动则是一个抽象概念，可以理解为Agent所有行为的总和。Agent的一切都是和这个环境相关联的：包括Agent的感知（输入）、大脑（内部处理）和行动（输出）。Agent也是环境的一部分，Agent的行动改变环境时，也可以改变Agent自身。

Agent的“运动”被大致分为：感知、内部处理和行动三个部分。对应架构中Perception、Brain和Action三个模块。Perception模块负责感知和处理来自外部环境的多模态信息；Brain模块负责记忆、思考和决策等内在任务；Action模块负责工具执行等。

Agent的内部信息通路是Perception->Brain->Action，而信息通路的设计本身也是Agent的一部分。完整的链路是：Agent通过一轮或者多轮“输入->处理->输出”来完成一个任务，任务是否完成由外界或内部的反馈信息来确定。

## Agent工作从哪些方面入手？

LLM Agents目前工作围绕以下三个方面展开：
* LLM Agent的整体框架
	* Agent的框架设计
	* 提升Agent能力的方法
* LLM Agent的应用场景
	* 社科
	* 自然科学
	* 工程
* LLM Agent评价方法


# Agent模块
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262000468.png)

LLM 本身具有很强的能力，需要通过设计合理的agent架构最大化agent的能力
目前的统一框架通常包括四个模块
* Profile模块主要识别Agent的角色
* Memory模块和Plan模块主要为了Agent能在动态变化的环境中具有总结过去和计划未来的能力
* Action模块是把Agent的决策转化为具体的行为

## Profile模块

Profile模块主要是为了表示Agent的角色，通常通过在prompt中写入影响LLM的行为。Agent Profiles可以包括age、gender、career、人设、社会信息和其他agent的关系等。这部分设计主要受到应用场景的影响。
主要有三种方式：
1. 手写设计
2. LLM生成
3. Dataset alignment

### Handcrafting Method

优点：比较灵活，可以为agent设计各种profile
缺点：labor-intensive，特别是agent数据量比较大的时候

1. Generative Agent：通过手写agent的信息描述agent的功能（name，objectives and relationships with other agents）

### LLM-Generation Method

Agent的Profiles通过LLM自动生成，需要对LLM设置生成的rules、composition and attributes。也可以设置seed examples作为few shot

优点：可以批量生产很多的profile
缺点：精确度比较难把控

1. RecAgent：首先通过手动制作Agent的背景（如年龄、性别、个人特征和电影偏好）为少数Agent创建种子档案。然后利用 ChatGPT 根据种子信息生成更多Agent档案。

### Dataset Alignment Method

profile 信息来自真实世界的datasets

1. Out of One, Many: Using Language Models to Simulate Human Samples：根据美国全国选举研究 (ANES) 参与者的人口统计背景（例如种族/民族、性别、年龄和居住州）为 GPT-3 分配角色。

## Memory 模块

Memory中存储了过去的感知和交互信息，可以借助这些信息促进后续未来的决策。Memory的主要功能包括：积累经验，自我进化，以更加一致、合理和有效的方式行事

### Memory Structures

Memory的设计借鉴了人类记忆模块的原理包括short-term memory和long-term memory
* short-term memory：prompt context中的信息
* long-term memory：向量库中的信息，可以通过query检索获取信息

#### Unified Memory

通常指的是short-term memory 通过 in-context learning直接将memory信息写入prompt当中。但这种方法受限于LLM的输入长度和LLM的长文本处理能力，很难将所有memory信息都集成到prompt中，并且过长的输入也会导致agent性能的衰退。

1. RLP：对话agent，直接将conversation存在in-context信息中
2. SayPlan：制定计划的Agent，将环境交互的场景graphs和环境的Feedback存在Memory中指导action

#### Hybrid Memory

同时使用short-term和long-term memory。short-term memory用于存储最近的感知信息，long-term memory用于存储整体的重要信息。在实际应用中short和long的结合可以增强agent的推理能力，对于过去成功经验的收集也可以帮助agent实现更多复杂的任务。

1. Generative Agent：short-term memory存储agent当前状态，long-term memory存储agent过去的action和thoughts，可以通过当前event retrieved增强当前决策

### Memory Format

memory的分类也可以根据memory的存储格式来分，不同的存储格式可以用于不同的场景。这些存储方式不是互斥的，很多agent会有多种存储方式

#### Natural Language

直接使用自然语言存储的memory信息表达的流程，容易理解也可以包含语义信息

1. Reflexion：直接在sliding window中存储Feedback

#### Embeddings

把memory信息encode成embedding向量，可以通过memory retrieval读取

1. MemoryBank：通过向量库存储reference plan，通过embeddings match retrieval读取复用 reference plan

Databases：可以让LLM生成SQL queries访问DB检索信息
Structured lists：memory信息可以被作为list等结构化数据存储

### Memory Operations

#### Memory Reading

Memory Reading的主要目标是从Memory中抽取重要的信息，增强agent的action。
Memory Reading的关键是如何从history action中抽取最有用的信息。通常有三种可以评价抽取信息的标准：recency，relevance和importance。检索过程可以通过以下函数表示

#### Memory Writing

Memory writing是为了将感知到的环境信息存入memory，为之后的retrieval提供基础知识。在做memory writing时需要注意两个关键点

1. Memory duplicated：需要解决新存入的信息和已存储信息的冗余问题
	GPT4：相同的子任务的解决方法会存入一个list，如果list长度超过5，所有解决方案将被LLM压缩成一个方案，然后重新生成一个方案存入memory
2. Memory overflow：需要解决存储快满时如何删除信息的问题
	RET-LLM：新的覆盖旧的（FIFO）

#### Memory Reflection

Memory Reflection主要目的是通过对memory中的内容总结，摘要提取出high-level信息

GITM：action成功执行子任务之后，任务信息会被存到list中，当list包括的超过五条信息，agent会总结摘要生成新的内容替换之前的
ExpeL：介绍了两种获取reflection的方法。第一种是通过对比成功的和失败的tarjection。第二种是agent通过学习收集成功的tarjection获取经验

## Plan模块

Plan是LLM Agents最关键的能力之一，需要复杂的理解、推理和决策过程。尽管规划是一个抽象概念，但规划任务一般的公式可以描述如下。

给定时间步骤t，环境表示为E，动作空间为A，任务目标为g，步骤t处的动作 $a_t \in A$ 。规划过程可以表示生成一系列动作：
$$p=(a_0,a_1,\cdots,a_t)=plan(E,g;\Theta,P)$$

对于LLM的agent plan可以分为5种重要类别：
1. 任务分解
2. 多计划选择
3. 外部模块辅助规划
4. 反思和细化
5. 记忆增强规划
![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202411242111309.png)

还有一种分类方法是：
1. Planning without Feedback
2. Planning with Feedback

Planning without Feedback实现起来比较简单，但也只适用于简单的任务，比如该任务需要的步长很短

不引入Feedback可能出现两个问题：
* 从开始就生成的plan没考虑很多复杂的条件，这就会使最开始设计的计划频频出错
* 执行的过程中计划尝尝会被外界环境的变化而修改
如果需要引入Feedback，需要设计合理的Feedback机制，否则会导致处理复杂问题时Feedback变得非常长。外界的Feedback可以来自环境、人和模型自身

### 任务分解

现实场景任务通常是复杂多变的，因此只通过一步规划过程解决复杂任务是非常困难有挑战性的。
因此需要对任务进行分解，分解成几个更简单的子任务，类似分而治之的算法策略。任务分解通常涉及两个关键步骤：
1. 分解复杂任务
2. 规划子任务
任务分解方法可以分为两大类：分解优先和交叉分解
![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202411242123497.png)

#### 分解优先

分解优先方法首先将任务分解成子目标，然后依次规划每个子目标
1. HuggingGPT: 利用Huggingface Hub中各种多模态模型构建用于多模态任务的agent，能处理各种多模态任务。其中LLM当做控制器，负责分解任务、选择模型并生成最终Response
2. Plan-and-Solve: 改进Zero-shot COT，将原来的 "Let's think step by step" 拆分成2步prompt "Let's first devise a plan" 和 "Let's carry out the plan"。这个zero shot方法在数学推理、常识推理和符号推理测试上取得了比较好效果。
3. ProgPrompt: 将任务的自然语言描述转化成Code问题。通过Code将Agent的动作空间和环境中的对象符号化，每个动作形式化成一个函数，每个对象表示成一个变量。任务规划转换成函数生成，Agent执行任务，以函数调用的形式生成规划，然后逐步执行。

#### 交叉分解

交叉分解涉及交叉的任务分解和子任务规划，每个分解仅显示当前状态下的一个或两个子任务
1. CoT：Chain-of-Thought通过一些构建的trail来指导LLM推理复杂问题，利用LLM推理能力进行任务分解。并且提出了Zero-shot prompt "Let's think step by step"
2. ReAct：没有ReAct之前，Reasoning和Acting是分割开来的。（Environmental Feedback）
	reasoning：CoT prompt没有交互能力，无法获得新的信息，会有幻觉
	acting：action plan generation（传统的RL Agent）只有和世界交互能力

	例子：让小朋友帮忙去厨房打酱油，告诉小朋友step by step (COT)
	1. 先看厨房灶台上有没有
	2. 再拉开灶台旁边的抽屉看看有没有
	3. 最后再看抽油烟机旁边的柜子里有没有

	没有ReAct的情况：
	> 不管在第几步没找到酱油，小朋友都会把这几个地方检查一遍 (Action)

	有ReAct的情况：
	> Action1: 先看厨房灶台上有没有
	> Observation1: 灶台上没有酱油，执行下一步
	> Action2: 再拉开灶台旁边的抽屉看看有没有
	> Observation2: 抽屉里有酱油
	> Action3: 把酱油拿出来

	```plaintext
	Question:
	Thought:
	Action:
	Observation:
	Thought:
	Action:
	Observation:
	```
	提升LLM Agent的Actions with verbal reasoning能力，也就是每次行动后都有一个Observation，做了啥是不是完成了。这更像是让LLM Agent维持了一个短期记忆。

	Reason without Observation
	* 相对 ReAct 中的 Observation 来说的，ReAct 提示词结构是 Thought -> Action -> Observation，Reason without Observation 是把 Observation 去掉了。但实际上 Reason without Observation 只是将 Observation 隐式嵌入到下一个执行单元中
3. PAL：PAL利用LLM的Code能力，指导LLM在推理过程中生成代码，利用代码解释器执行代码获得解决方案。主要用于提升Agent解决数学和符号推理问题能力。
4. PoT：Program-of-Thought将推理过程完全形式化为Code

#### 总结

分解优先：子任务和原始任务之间建立更强的相关性，从而降低任务遗忘和幻觉的风险，由于子任务是在开始时预先确定的，因此需要额外的调整机制，否则某个步骤中的一个错误将导致失败。
交叉分解：根据环境反馈动态调整分解，提高容错能力。然而，对于复杂的任务，过长的轨迹可能会导致 LLM 出现幻觉，在后续的子任务和子规划中偏离原始目标。

挑战：
1. 任务分解带来的额外开销：将一个任务分解为多个子任务需要更多的推理和生成，从而产生额外的时间和计算成本。
2. 如果任务是高度复杂任务，将任务分解成多个子任务，规划受到LLM上下文长度的限制，会导致规划被遗忘

### 多计划选择

由于任务复杂性和LLM的不确定性，LLM Agents为给定任务生成的计划可能多种多样。尽管LLM有推理能力，但是LLM生成的单个计划很可能不是最优的，甚至是不可行的。更合适的方式也就是多计划选择，主要包括两步：多计划生成和最优计划选择

1. Self-consistency：复杂问题的解决方案很少是唯一的，与生成单一路径CoT相比，Self-consistency通过采样策略获得多条不同推理路径，并且采用朴素的投票策略，将获得最多票数的方案视为最优选择。
2. Tree-of-Thought：ToT提出了两种生成计划策略：采样和提议。采样策略和Self-consistency一致；提议策略明确指示LLM通过prompt中的few shots生成各种计划。因为是树结构，ToT支持树搜索算法，例如BFS和DFS。在选择扩展节点（为了寻找最优解而进一步分析的节点），使用LLM评估多个操作并选择最佳操作。

类似的还可以使用蒙特卡洛树搜索算法来进行搜索
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262001410.png)

#### 总结

多计划选择：具有可扩展性，可以搜索空间中更广泛地探索潜在解决方案。
增加的计算需求，特别是对于具有大量标记计数或计算的模型。特别是在资源限制是一个重要因素的情况下，例如在线服务。
依赖 LLM 来评估计划，由于 LLM 在排名任务中的表现仍不确定，因此需要进一步验证和微调其在这种特定情况下的能力。LLM 的随机性增加了选择的随机性，可能会影响所选计划的一致性和可靠性。

### 反思和改进

反思和改进增强了LLM Agents规划的容错和纠错能力。由于存在幻觉问题和复杂问题的推理能力不足，LLM Agents可能在规划过程中犯错，并且因为feedback有限陷入思维循环。反思和总结失败有助于Agent纠错并在后续尝试中摆脱这种循环。

1. Self-refine：采样生成、反馈和改进的迭代过程。每次生成之后，LLM都会为计划生成反馈，根据反馈进行调整（Model Feedback）
2. Reflexion：Reflection类似于一个是学生（Generator）来完成项目，老师（Reflector）来批改给出建议，学生再根据批改建议修改反复。
	Reflexion是Reflection的升级版，是运用了强化学习的思路。和Reflection相比，引入外部数据来评估回答是否准确，强制生成Response中多余和缺失的方面。（Model Feedback）

	Reflexion框架包含四个组成部分：
	![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262002980.png)

	- Actor: Actor由LLM担任，主要工作是基于当前环境生成下一步的动作。
    - Evaluator: Evaluator主要工作是衡量Actor生成结果的质量。就像强化学习中的Reward函数对Actor的执行结果进行打分。
    - Self-reflexion：Self-reflexion一般由LLM担任，是Reflexion框架中最重要的部分。它能结合离散的reward信号(如success/fail)、trajectory等生成具体且详细语言反馈信号，这种反馈信号会储存在Memory中，启发下一次实验的Actor执行动作。相比reward分数，这种语言反馈信号储存更丰富的信息，例如在代码生成任务中，Reward只会告诉你任务是失败还是成功，但是Self-reflexion会告诉你哪一步错了，错误的原因是什么等。
    - Memory：分为短期记忆(short-term)和长期记忆(long-term)。在一次实验中的上下文称为短期记忆，多次试验中Self-reflexion的结果称为长期记忆。类比人类思考过程，在推理阶段Actor会不仅会利用短期记忆，还会结合长期记忆中存储的重要细节，这是Reflexion框架能取得效果的关键。

    Reflexion的执行流程：
    ![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262002074.png)
    Reflexion是一个迭代过程，Actor产生行动，Evaluator对Actor的行动做出评价，Self-Rflexion基于行动和评价形成反思，并将反思结果存储到长期记忆中，直到Actor执行的结果达到目标效果。
    - Step 1 Actor生成运行的初始 trajectory
    - Step 2 使用Evaluate评价效果输出评价结果，把评价结果存入mem
    - Step 3 循环进行通过循环迭代tarjectory直到达到最大次数

#### 总结

反思策略和强化学习原理相似，其中agent扮演决策者的角色，环境反馈触发策略网络的更新。然而，不同于强化学习通过修改模型参数实现更新，LLM Agents更新是通过LLM本身的自我反思进行的，最终以文本反馈结束。这些文本反馈可以作为长期和短期记忆，通过prompt影响agent后续的输出。
但文本形式更新收敛目前缺乏保证的证明，无法证明持续反思最终可以引导LLM Agents实现指定的目标。

### 评估

1. 交互式游戏环境
	游戏环境可以根据agent的动作提供实时的多模态反馈，包括文本和视觉反馈。目前，使用最广泛的游戏环境是Minecraft，其中agent需要收集材料来创建工具以获得更多奖励。agent创建的工具数量通常用作评估指标。
	另一个流行的类别是基于文本的交互式环境，例如ALFWorld、ScienceWorld等，其中agent位于用自然语言描述的环境中，具有有限的动作和位置。成功率或获得的奖励通常用作评估指标。与 Minecraft 相比，这些基于文本的交互环境往往更加简单，反馈直接，可操作性更少。（但看case也更容易陷入循环）
2. 交互式检索环境
	交互式检索环境模拟了人类在现实生活中进行的信息检索和推理的过程。在这些环境中，agent通常被允许与搜索引擎和其他 Web 服务进行交互，使用诸如搜索关键字或执行点击、前进和后退等操作来获取更多信息，从而获得回答问题或完成信息检索任务。 常用的检索环境包括基于维基百科引擎的问答任务（例如HotPotQA 和Fever ）和网页浏览任务以查找特定信息，包括WebShop，Mind2Web 和WebArena。任务成功率通常用作衡量标准。
3. 交互式编程环境
	交互式编程环境模拟程序员与计算机之间的交互，测试agent在解决计算机相关问题时的规划能力。在这些环境中，agent需要通过编写代码或指令与计算机交互以解决问题。他们将收到各种反馈，包括编译和运行时错误消息以及执行结果。流行的交互式编程环境涉及与操作系统、数据库等相关的问题，例如 Agent Bench、MiniWoB++。 大多数现有的交互式环境缺乏细粒度的评估，其中性能主要通过最终成功率来评估。此外，与现实世界中通常有多条路径来完成任务的情况不同，由于注释成本高，大多数模拟环境中通常只有一条“黄金”路径。

# MultiAgents

## Single-Agent VS Multi-Agent Systems

Single-Agent系统的构建集中于形式化它们的内部机制和与外部环境的互动。

LLM-MA系统强调多样化的Agents资料，Agents的互动，以及集体决策过程。从这个角度看，更动态和复杂的任务可以通过Agents的协作来解决，每个Agent都配备了独特的策略和行为，并参与与其他agents的沟通。
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262003816.png)

## Agents Communication

在LLM-MA系统中Agents之间的通信是支持集体智能的关键基础设施。从三个角度剖析Agents通信：
1. 通信范式：Agents之间的交互风格和方法
2. 通信结构：Agents系统内通信网络的组织和架构
3. Agents之间交换的通信内容

### Communication Paradigms

- 当前的LLM-MA系统主要采用三种通信范式：合作，辩论和竞争
    - 合作型Agents共同努力实现共享的目标或目标，通常通过交换信息来增强集体解决方案
    - 当Agents进行争论性的交互时，就会采用辩论范式，他们会提出并捍卫自己的观点或解决方案，并批评他人的观点。这种范式非常适合达成共识或更加精细的解决方案
    - 竞争型Agents努力实现自己的目标，这些目标可能与其他Agents的目标冲突

### Communication Structure
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262003777.png)
- LLM-MA系统中四种典型的通信结构
    - 分层通信是分层结构的，每个层次的Agents具有不同的角色并主要在其层次内或与相邻层次进行交互。一个称为动态LLM-Agent网络（DyLAN）的框架，该框架在多层前馈网络中组织Agents。这种设置便于进行动态交互，包括推理时Agents选择和早停机制等功能，这些功能共同提高了Agents合作的效率
    - 去中心化通信在点对点网络上运行，Agents直接相互通信，这是世界模拟应用程序常用的结构
    - 集中式通信涉及一个中心Agent或一组中心Agents协调系统的通信，其他Agents主要通过这个中心节点进行交互
    - 为了提高通信效率，提出了共享消息池。这种通信结构维持一个共享消息池，Agent在其中发布消息并根据其配置文件订阅相关消息，从而提高通信效率

### Communication Content

- 在LLM-MA系统中，通信内容通常以文本的形式出现。具体内容差异很大，取决于特定的应用程序。在软件开发中，Agents可能会互相讨论代码段。在像狼人杀这样的游戏模拟中，Agents可能会讨论他们的分析、怀疑或策略

## Agents Capabilities Acquisition

Agents能力获取是LLM-MA中的关键过程，使Agents能够动态地学习和发展。在这个背景下，有两个基本概念：
1. Agents应该从中学习以提高他们能力的反馈类型
2. Agents调整自己以有效解决复杂问题的策略

### Feedback

- 反馈是Agents关于其行为结果接收的关键信息，帮助Agents了解其行为可能的影响，并适应复杂和动态的问题。在大多数研究中，向Agents提供的反馈的格式是文本的。
- 根据Agent接收此反馈的来源，反馈可以被归类为四种类型
    - 来自环境的反馈:来自现实世界环境或虚拟环境.这在大多数LLM-MA的问题解决场景中都很常见，包括软件开发（Agents从代码解释器获取反馈）和具体化的Agents系统（机器人从现实世界或模拟环境获取反馈）
    - 来自Agents互动的反馈,反馈来自其他Agents的判断或来自Agents的通信。它在像科学辩论这样的问题解决场景中很常见，Agents通过通信学习批判性地评估和改进结论。在像游戏模拟这样的世界模拟场景中，Agents学会根据其他Agents之间的以前的互动来改善策略
    - 反馈直接来自人类，对于将Agents系统与人类的价值观和偏好相对齐是至关重要的。这种反馈在大多数“Human-in-the-loop”的应用中
    - 无。在某些情况下，没有反馈提供给Agents。这通常发生在世界模拟工作中，这些工作重点是分析模拟结果，而不是Agents的规划能力。在这样的场景中，比如传播模拟，重点是结果分析，因此，反馈不是系统的组成部分

### Agents Adjustment to Complex Problems

- 为了提高他们的能力，LLM-MA系统中的Agents可以通过三种主要的解决方案进行适应。
    - Memory：大多数LLM-MA系统利用一个Memory模块让Agents调整他们的行为。Agents在他们的Memory中存储前一次交互和反馈的信息。在执行行动时，他们可以检索相关的、有价值的memory，特别是那些包含成功行动的memory，这些memory用于完成类似的过去目标。这个过程有助于增强他们当前的行动。
    - Self-Evolution：与Memory基础解决方案中仅依赖历史记录来决定后续行动的情况不同，Agents可以通过修改自己，如改变他们的初始目标和规划策略，基于反馈或通信日志对自己进行训练，来动态地self-Evolution。
        1. 提出了一个自我控制循环过程，允许Agents系统中的每个Agent自我管理和自我适应动态环境，从而提高Agents的合作效率
        2. ProAgent，它预测队友的决策，并根据Agents之间的通信日志动态调整每个Agent的策略，促进相互理解并提高协同规划能力
        3. 讨论了一个通过通信学习（LTC）范式，使用Agents的通信日志生成数据集来训练或微调LLM。LTC通过与他们的环境和其他Agents的交互，使Agents能够持续适应和改进，打破了在上下文学习或监督微调的限制，这些限制并未充分利用在与环境和外部工具的交互中收到的反馈进行持续训练
         Self-Evolution使Agents能够在他们的个人资料或目标中自主调整，而不仅仅是从历史交互中学习
    - Dynamic Generation：在一些场景中，系统可以在其操作过程中实时生成新的Agent。这种能力使系统能够有效地扩展和适应，因为它可以引入那些专门设计来应对当前需求和挑战的Agent

# Workflow + Agent

LLM Agents被设计通过迭代规划和行动来完成复杂任务。然而，当缺乏专业知识密集型任务的具体知识时，Agent容易出现幻觉生成错误的规划。为了解决这样的问题，给Agent Workflow作为参考来提高规划可靠性。

## Workflow形式
目前Workflow的格式主要有以下几种

| workflow格式 | Text                                    | Code                    | FlowChart（工作流程图）                    | PseudoCode （text+flowchart） |
| ---------- | --------------------------------------- | ----------------------- | ----------------------------------- | --------------------------- |
| 优点         | 表达灵活，容易编辑                               | 表达精确                    | 表达精确，用户友好，容易编辑                      | 步骤精确，用户友好                   |
| 缺点         | token量多                                 | 有编程门槛                   | 需要额外定义状态，有编程门槛                      | 需要先翻译，然后再执行；复杂场景模型执行容易乱步    |
| 性能表现       | 在对话轮次上要差于Code和FlowChart；但在Session粒度上会更好 | 对于LLM自身要求比较高，不同LLM方差比较大 | 在对话轮次上优于前两种，但在session粒度上对LLM自身能力有要求 | -                           |

### FlowBench
关于workflow的benchmark，提供了对于workflow格式化的建议以及示例，并且提出集成的benchmark。作者测评了以上三种(text, code, flowchart)在benchmark上的表现

#### 数据集构造
数据集构造策略：场景搜集、workflow设计、会话生成
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411261959194.png)

- 场景搜集：搜集了涵盖6个领域对应的场景，人设。(Customer Service、Personal Assistant、E-tail Recommendation、Travel&Transportation、Logistics Solutions、Robotic Process Automation)
- workflow设计：首先从知识库(wikihow)、工作流网站(Zapier)中搜索相关流程。然后通过人工标注出text格式，最后通过GPT-4转化为其他格式的workflow。
- 会话数据生成：利用GPT-4生成多样的场景(用户背景，用户目标，响应风格...)，然后利用人工和GPT-4交叉生成、验证数据。

#### 评估方式
主要有轮次粒度和会话粒度的的评估方式。

- Tool invocation：工具的名称和参数都被正常使用
- Response quality：GPT-4评分，从正确性，帮助性，友好程度评分(10)
- Success rates：会话完结的比例。
- Task Progress：平均每个会话完成的程度

#### 实验结果
- 不同知识格式的效果在不同的环境下存在差异。Code、FlowChart在较弱的llm上效果较差，可能是因为复杂的符号表达式阻碍了信息的传递。另一方面，文本格式在不同的llm上继续表现良好。
- 在作者的实验中，FlowChart在轮次级别的表现往往优于其他两种格式，但是在整个对话粒度上，基本上和text格式持平
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262007042.png)
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262008337.png)

domain-wise
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262009864.png)
- 在Customer Service、Personal Assistant比较偏向自然语言交互的场景中，有无workflow影响不是很大。但是在后续需要逻辑推理、额外的工作知识的场景，LLM对工作流的逐渐依赖。
- Code和FlowChart对于逻辑性比较强、规划性比较强的任务具有很好的适应性，但是Code和FlowChart对于LLM的能力要求较高。

### CoRE
基于workflow，code，natural language三种形式，提出CoRE语言，一种对workflow进行伪代码编程的语法。
在执行CoRE语言的时候，需要使用LLM作为Interpreter，首先对CoRE语言进行翻译，然后和其他内容拼接，作为执行模型的输入。CORE语言由Step Name，Step Type(Process、Decision、Terminal)、Step Instruction、Step Connection。四种组件组成，各个组件由:::分割开
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262017305.png)

#### 实现
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262019124.png)

CoRE语言的每一步都需要基于LLM的Interpreter。在每一Step中，需要执行以下四步。
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262021273.png)
- Interpreter首先获取到当前步骤的有用信息
- 将相关信息拼接构建prompt（Task Description(对于整个程序的描述，基本上是基于用户的query)，Current Progress(对当前为止已经做过的和决定过的总结)，Observation(使用工具时才有，工具执行结果)，Current Instruction(当前要执行的操作)
- 基于prompt生成response(或者使用工具获取执行结果)
- 基于当前步骤以及输出结果决定下一个Step

#### 效果
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411262023595.png)
- CoRE语言显著优于Zero-shot和CoT，但是在一些任务上，不如Few-shot。作者认为是他们没有很好的给定输出格式的限制，而few-shot的方式则隐式的给出了输出格式。
- CoRE语言对于LLM的自身能力要求也比较高，GPT-4的效果就明显高于Mixtral

## Workflow生成
目前主要还是基于ICT的方法

AutoFlow：基于微调的方法和基于上下文的方法。基于微调的方法通过调整llm的参数，为特定的任务和领域定制了工作流生成过程。基于上下文的方法利用上下文信息来指导生成过程，而不需要进行广泛的微调，这使得它同时适用于开源和闭源的llm

FlowMind: 和RAG不一样，让LLM在给定知识的条件下，生成code形式workflow，在一定程度上保证数据隐私性。并且引入用户反馈机制，辅助LLM对生成的workflow进行改写修正

# 挑战

### 幻觉

幻觉问题是LLMs和单一LLM基础Agent系统中的一个重大挑战。它指的是模型生成事实上不正确的文本的现象 。
然而，在Agents设置中，这个问题增加了一层复杂性。在这样的情况下，一个Agent的幻觉可以产生连锁反应。这是由于Agents系统的相互连接性，其中一个Agent的错误信息可能被网络中的其他人接受并进一步传播。
它涉及的不仅是纠正个别Agent的不准确性，而且还涉及管理Agents间的信息流动，以防止这些不准确性在整个系统中的传播。

## 交互环境

大多数先前关于LLM-MA的工作都集中在基于文本的环境中，擅长处理和生成文本。然而，在多模态设置中存在显著的缺乏，Agents将与多感官输入进行互动并解释数据，并生成多个输出，如图像，音频，视频和物理动作。将LLM集成到多模态环境中带来了额外的挑战，例如处理多样化的数据类型并使Agent能够理解彼此并回应超过仅仅文本信息的更多东西。

## 能力提升

在传统的Agents系统中，Agent通常使用强化学习从离线训练数据集中学习。然而，LLM-MA系统主要从即时反馈中学习，比如与环境或人类的交互。这种学习风格需要一个可靠的交互环境，而为许多任务设计这样一个交互环境会很棘手，限制了LLM-MA系统的可扩展性。

对于classification、regression这样的supervised learning 任务，生成一步决定就完成任务了。这样的agent与planning无关，不涉及agent能力提升。
对于sequential decision making这样的多步序列决策来说，每一步的决策，每一个action，都可能影响以后的决策，很可能影响整个决策的质量。一步一步生成action很有可能导致不是最优解。
多步序列决策sequential decision making需要通盘考虑，需要有planning、强化学习等算法，制定学习策略。目前大部分LLMs没有专门训练多步决策。另外，多步决策或许也不一定用LLM

## Scaling Up

LLM-MA系统由多个单独的LLM-based agent组成，对于agent的数量带来了显著的可扩展性挑战。从计算复杂性的角度看，每一个基于大型语言模型（如GPT-4）的LLM-based agent都需要大量的计算能力和内存。在LLM-MA系统中增加这些agent的数量会显著增加资源需求。在计算资源有限的场景中，开发这些LLM-MA系统将会非常具有挑战性。

此外，随着LLM-MA系统中Agent数量的增加，额外的复杂性和研究机会开始出现，尤其是在像有效Agents协调、通信以及理解Agents规模法则等领域。
- 随着更多基于LLM的Agent出现，确保有效协调和通信的复杂性显著增加。
- 设计先进的Agents协调方法越来越重要。这些方法旨在优化Agents工作流，根据不同Agents量身定制的任务分配，以及跨Agents的通信模式，例如Agents之间的通信约束。有效的Agent协调能够促进Agent之间的和谐操作，最小化冲突和冗余。
- 探索和定义规定Agents系统行为和效率的规模法则，随着它们变得越来越大，仍然是一个重要的研究领域。这些方面强调了需要创新解决方案来优化LLM-MA系统，使它们既有效又资源高效

## 评测

目前Agent评测往往只是单任务是否完成评测，缺乏任务细粒度评测以及Multi Agent评测。并且在许多领域方向缺乏全面的评测集。

## 应用

目前Agent应用更多还是在角色扮演、代码（Cursor）等工作上，如何推广至更多开放世界领域并且建立对应仿真环境是需要进行考虑的。

# Reference

## Awesome
https://github.com/fr0gger/Awesome-GPT-Agents
https://github.com/e2b-dev/awesome-ai-agents
https://github.com/jun0wanan/awesome-large-multimodal-agents

## Survey

[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864)
[A survey on large language model based autonomous agents](https://journal.hep.com.cn/fcs/EN/article/downloadArticleFile.do?attachType=PDF&id=37552)
[Understanding the planning of LLM agents: A survey](https://arxiv.org/pdf/2402.02716)
[Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/pdf/2402.01680)
[AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION](https://arxiv.org/pdf/2401.03568)

## Profile
[Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/pdf/2307.02485)
[RecAgent: A Novel Simulation Paradigm for Recommender Systems](https://www.researchgate.net/publication/371311704_RecAgent_A_Novel_Simulation_Paradigm_for_Recommender_Systems)
[Out of One, Many: Using Language Models to Simulate Human Samples](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/035D7C8A55B237942FB6DBAD7CAA4E49/S1047198723000025a.pdf/out-of-one-many-using-language-models-to-simulate-human-samples.pdf)

## Memory
[Reflective Linguistic Programming (RLP): A Stepping Stone in Socially-Aware AGI (SocialAGI)](https://arxiv.org/pdf/2305.12647)
[SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://arxiv.org/pdf/2307.06135)
[Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/pdf/2307.02485)
[MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/pdf/2305.10250)
[GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774)
[RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/pdf/2305.14322)
[Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory](https://arxiv.org/pdf/2305.17144)
[ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/pdf/2308.10144)

## Plan
### Task Decomposition
[HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/pdf/2303.17580)
[Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/pdf/2305.04091)
[PROGRESSIVE PROMPTS: CONTINUAL LEARNING FOR LANGUAGE MODELS](https://arxiv.org/pdf/2301.12314)
[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
[ReAct: Synergizing Reasoning And Acting In Language Models](https://arxiv.org/pdf/2210.03629)
[ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/pdf/2305.18323)
[PAL: Program-aided Language Models](https://arxiv.org/pdf/2211.10435)
[Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://arxiv.org/pdf/2211.12588)

### Multi-plan Selection
[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/pdf/2203.11171)
[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)

### Reflection&Refinement
[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/pdf/2303.17651)
[Reflection Agents](https://blog.langchain.dev/reflection-agents/)
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)

## Workflow
[FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents](https://arxiv.org/pdf/2406.14884)
[CoRE: LLM as Interpreter for Natural Language Programming, Pseudo-Code Programming, and Flow Programming of AI Agents](https://arxiv.org/pdf/2405.06907v1)
[AutoFlow: Automated Workflow Generation for Large Language Model Agents](https://web3.arxiv.org/pdf/2407.12821)
[FlowMind: Automatic Workflow Generation with LLMs](https://arxiv.org/pdf/2404.13050)
