---
tags:
  - llm
  - sft
  - lora
  - merge
date: 2025-01-11T19:09:00
---
# Model Merging Recipes

CodeBase:
[mergekit](https://github.com/arcee-ai/mergekit)

Papers:
[Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/pdf/2403.13187)
[TIES-MERGING: Resolving Interference When Merging Models](https://arxiv.org/pdf/2306.01708)

## Exp (WIP)

### TL;DR

1. 在base model上进行sft lora训练，生成对应adapter
2. 将生成的adapter合并到instruct model上
3. 将多个合并后的模型使用ties融合

### Highlight

1. 针对性补强模型能力，且其余能力损失少
2. 迭代快，可以快速得到当前数据集对模型的效果，减少混合配比等尝试成本

### Lowlight

1. 需要一个base model以及一个instruct model
2. 对于领域能力来说，lora学习的效果不如full sft，并且经过模型融合后，领域能力被进一步削弱

### Todo

1. 使用ifd + lora merge
2. 基于ct + 退火 + two stage sft 得到需要的instruct model
