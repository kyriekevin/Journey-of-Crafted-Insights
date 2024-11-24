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

Agent是“运动”在“环境”（Environment）中，环境描述了Agent所在的状态空间，运动则是一个抽象概念，可以理解为Agent所有行为的总和。Agent的一切都是和这个环境相关联的：包括Agent的感知（输入）、大脑（内部处理）和行动（输出）。Agent也是环境的一部分，Agent的行动改变环境时，也可以改变Agent自身。

Agent的“运动”被大致分为：感知、内部处理和行动三个部分。对应架构中Perception、Brain和Action三个模块。Perception模块负责感知和处理来自外部环境的多模态信息；Brain模块负责记忆、思考和决策等内在任务；Action模块负责工具执行等。

Agent的内部信息通路是Perception->Brain->Action，而信息通路的设计本身也是Agent的一部分。完整的链路是：Agent通过一轮或者多轮“输入->处理->输出”来完成一个任务，任务是否完成由外界或内部的反馈信息来确定。

## LLM Agents 涉及技术
1. RAG: 知识库导入、长期记忆支持、风格化个性化（用户信息）等
2. COT：问题的分解与推理
3. 符号推理：基于上下文，从一组备选选项中选择合适类别，根据需要填入相应参数，并进行格式化输出（Function Call和马尔科夫决策）往往在中间步骤中体现
4. 数据通路和行动框架：数据通路指的是Agent内部信息流动的机制；行动框架指的是Agent决策的算法和策略，定义Agent如何根据输入的数据和内部状态来选择行动。

# Agent 设计模式

## ReAct

没有ReAct之前，Reasoning和Acting是分割开来的

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

### Prompt Template
```
Question:
Thought:
Action:
Observation:

Thought:
Action:
Observation:
```

提升LLM Agent的Actions with verbal reasoning能力，也就是每次行动后都有一个Observation，做了啥是不是完成了。这更像是让LLM Agent维持了一个短期记忆。
## Plan and Solve

先有计划再执行，计划可能发生改变。适合需要对任务设计计划，并且在过程中计划可能发生改变。

### Prompt Template

## Reflection
Reflection类似于一个是学生（Generator）来完成项目，老师（Reflector）来批改给出建议，学生再根据批改建议修改反复

## Reflexion
Reflexion是Reflection的升级版，是运用了强化学习的思路。和Reflection相比，引入外部数据来评估回答是否准确，强制生成Response中多余和确实的方面。

### Prompt Template
会让大模型针对问题在回答前进行反思和批判性思考，反思包括有没有遗漏（missing）或者重复（superfluous），然后回答问题，回答之后再针对性的修改（revise）

* Reason without Observation
	* 相对 ReAct 中的 Observation 来说的，ReAct 提示词结构是 Thought -> Action -> Observation，Reason without Observation 是把 Observation 去掉了。但实际上 Reason without Observation 只是将 Observation 隐式嵌入到下一个执行单元中
* LLMCompiler
	* 通过任务编排使得计算更有效率，即通过并行function call来提高效率
* Language Agent Tree Search
	* Tree Search + ReAct + Plan&Solve,通过Tree Search进行Reward，融入Reflection来拿到结果
* Self-Discover
	* 让大模型在更小粒度上task本身进行反思。Plan&Solve反思task是否需要补充，Self-Discover反思task本身
* Storm
	* Agent利用外部工具搜索生成大纲，然后再生成大纲里的每部分内容

# Reference
[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864)
[ReAct: Synergizing Reasoning Ang Acting In Language Models](https://arxiv.org/pdf/2210.03629)
[Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://arxiv.org/pdf/2305.04091)
<div id="ref3"></div>
[Reflection Agents](https://blog.langchain.dev/reflection-agents/)
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)
[ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/pdf/2305.18323)
[An LLM Compiler for Parallel Function Calling](https://arxiv.org/pdf/2312.04511)
[Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406)
[Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/pdf/2402.03620)
[Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/pdf/2402.14207)
