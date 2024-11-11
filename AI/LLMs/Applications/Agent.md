---
tags:
  - llm
  - AI
date: 2024-11-04T10:16:00
---

# Agent 是什么？

Agent 是一个经典概念，强化学习入门时候就多次提及 agent，代指环境交互的实体。 学界对 Agent 有更 high-level 的期待，例如维基百科对 Intelligent Agent的定义：

> In intelligence and artificial intelligence, an intelligent agent (IA) is an agent that perceives its environment, takes actions autonomously in order to achieve goals, and may improve its performance with learning or acquiring knowledge. An intelligent agent may be simple or complex: A thermostat or other control system is considered an example of an intelligent agent, as is a human being, as is any system that meets the definition, such as a firm, a state or a biome.

Agent 核心能力是完成任务（achieve goals）、获取知识（acquiring knowledge）和进化（improve）

## LLM Agents 是什么？

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

参考复旦NLP组综述提出LLM Agents框架 [<sup>1</sup>](#ref1)

![alt text](img/1.png)

# Agent 设计模式

## ReAct [<sup>2</sup>](#ref2)

![alt text](img/2.png)

### 原理

没有ReAct之前，Reasoning和Acting是分割开来的

例子：让小朋友帮忙去厨房打酱油，告诉小朋友step by step (COT)
1. 先看厨房灶台上有没有
2. 再拉开灶台旁边的抽屉看看有没有
3. 最后再看抽油烟机旁边的柜子里有没有

没有ReAct的情况：
> 不管在第几步没找到酱油，小朋友都会把这几个地方检查一遍 (Action)

有ReAct的情况：
> Action1: 先看厨房灶台上有没有
Observation1: 灶台上没有酱油，执行下一步
Action2: 再拉开灶台旁边的抽屉看看有没有
Observation2: 抽屉里有酱油
Action3: 把酱油拿出来

## Plan and Solve [<sup>3</sup>](#ref3)

先有计划再执行，计划可能发生改变

## Reason without Observation [<sup>4</sup>](#ref4)

Reason without Observation 是相对 ReAct 中的 Observation 来说的，ReAct 提示词结构是 Thought -> Action -> Observation，Reason without Observation 是把 Observation 去掉了。但实际上 Reason without Observation 只是将 Observation 隐式嵌入到下一个执行单元中

例子：常见审批流程是有前置依赖，环环相扣的
1. 从A部门拿到a文件
2. 从B部门提交a文件办理b文件
3. 从C部门提交b文件办理c文件

2，3步中B C部门对a b文件检查就是一类Observation

## LLMCompiler [<sup>5</sup>](#ref5)

通过任务编排使得计算更有效率，即通过并行function call来提高效率

例子：用户查询淘宝 京东 抖音 三个平台同一物品哪个便宜，同时搜索三个平台上物品价格，最后合并得出结果


# Reference

<div id="ref1"></div>

[1] [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864)

<div id="ref2"></div>

[2] [ReAct: Synergizing Reasoning Ang Acting In Language Models](https://arxiv.org/pdf/2210.03629)

<div id="ref3"></div>

<div id="ref4"></div>
