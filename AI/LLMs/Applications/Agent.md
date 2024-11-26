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



# Brain模块
## Plan

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
2. ReAct：没有ReAct之前，Reasoning和Acting是分割开来的。
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

#### 总结

多计划选择：具有可扩展性，可以搜索空间中更广泛地探索潜在解决方案。
增加的计算需求，特别是对于具有大量标记计数或计算的模型。特别是在资源限制是一个重要因素的情况下，例如在线服务。
依赖 LLM 来评估计划，由于 LLM 在排名任务中的表现仍不确定，因此需要进一步验证和微调其在这种特定情况下的能力。LLM 的随机性增加了选择的随机性，可能会影响所选计划的一致性和可靠性。

### 反思和改进

反思和改进增强了LLM Agents规划的容错和纠错能力。由于存在幻觉问题和复杂问题的推理能力不足，LLM Agents可能在规划过程中犯错，并且因为feedback有限陷入思维循环。反思和总结失败有助于Agent纠错并在后续尝试中摆脱这种循环。

1. Self-refine：采样生成、反馈和改进的迭代过程。每次生成之后，LLM都会为计划生成反馈，根据反馈进行调整
2. Reflexion：Reflection类似于一个是学生（Generator）来完成项目，老师（Reflector）来批改给出建议，学生再根据批改建议修改反复。
	Reflexion是Reflection的升级版，是运用了强化学习的思路。和Reflection相比，引入外部数据来评估回答是否准确，强制生成Response中多余和确实的方面。
	会让大模型针对问题在回答前进行反思和批判性思考，反思包括有没有遗漏（missing）或者重复（superfluous），然后回答问题，回答之后再针对性的修改（revise）

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

### 挑战

1. 幻觉
	规划过程中，LLM经常实现幻觉，导致计划不合理、不忠于任务提示或者无法遵循复杂指令
2. 生成计划可行性
	LLM还是通过大数据优化下一个token概率，这样的方法难以遵循复杂的约束，尤其是LLM遇到不常见的约束
3. 环境
	LLM最初还是处理文本输入设计，但是世界环境还有反馈往往是多模态，往往很难用自然语言描述清楚
4. 评估
	现在的benchmark主要依赖任务的最终完成状态，缺乏细粒度的分步评估



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
<div id="ref3"></div>
[Reflection Agents](https://blog.langchain.dev/reflection-agents/)
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)

[An LLM Compiler for Parallel Function Calling](https://arxiv.org/pdf/2312.04511)
[Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406)
[Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/pdf/2402.03620)
[Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207)
