---
tags:
  - llm
  - infra
  - 910b
date: 2024-11-10T21:00:00
---

# Merlin基础知识
## 集群
集群：
CPU类型：arm
选择方式：
## nas挂载
由于是arm环境，ByteNas文件系统挂载会有问题，出现一直处于镜像启动状态
解决方法：需要ByteNas上把协议修改为NFSv4.1，fuse挂载不支持
> 改为NFSv4.1后无法挂载在开发机上，目前还是只能通过hdfs来传输文件

## 镜像
CANN（Compute Architecture for Neural Networks）是华为的专门包，也需要对齐版本
vllm需要Python 3.9+,Llama-Factory华为只对齐了Python3.9和3.10的版本，建议选第三个镜像

| 环境配置                                                | 镜像                                                                              | Trail                                                                                                                          |
| --------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Python: 3.8.18<br>Pytorch:2.1.0<br>Cann:8.0.RC2<br> | <br>hub.byted.org/ies_sc_train/ies_sc_910b_arm:2fa6ed0a43c6a150ad08d34b16acc1a3 | <br>[910b arm 1gpu](https://ml.bytedance.net/development/instance/jobs/cea7e4ef6c0c4605?tabState=task_config&trialId=35537755) |
| Python: 3.9.19<br>Pytorch:2.1.0<br>Cann:8.0.RC2     | hub.byted.org/ies_sc_train/ies_sc_910b_arm:fc1129f94663dbf395f1d30e6b81bdce     | [910b arm 1gpu](https://ml.bytedance.net/development/instance/jobs/4e4ae532294d92d0?tabState=task_config&trialId=32692568)     |
| Python: 3.9.19<br>Pytorch:2.3.1<br>Cann:8.0.T39     | hub.byted.org/ies_sc_train/ies_sc_910b_arm:8c0876a34efd7176cc3eb5c106e2bf8c     | [910b arm 1gpu](https://ml.bytedance.net/development/instance/jobs/08e8679571b16145?tabState=run_info&trialId=35641447)        |

# 910b基础知识
## Toolkit
## 监控工具
类似于`nvidia-smi`，910b需要使用`npu-smi info`命令进行查看
AI Core指标对应Nvidia的SM（通常50-60%就算比较高）
HBM-Usage指标对应Nvidia的显存利用率

## 代码测试
检查基础环境是否符合预期，直接使用python3执行下述代码。如果有问题需要查看环境变量等是否配置有问题。
```Python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch.rand([100, 100]).cuda() @ torch.rand([100, 100]).cuda()
```

## 代码适配
在代码中加入以下语句，可以将原来的cuda、nccl等底层实现替换成昇腾相关实现
```Python
import torch         # 在【最开始的】import torch后面加入下面两行代码

import torch_npu      # 注意：torch 和 torch_npu的版本是强对应的，不要更改torch版本，在安装依赖库时要特别注意
from torch_npu.contrib import transfer_to_npu # 执行替换操作
```
如还遇到问题可以尝试手动替换
```Python
.npu()
.to('npu')
```
一般可能是python版本和torch版本没对应上，不要修改镜像里python版本和torch、torch_npu版本！！！

# 具体使用

## Train

### Llama-Factory

#### 基本信息
repo：
trail：
config：

#### 使用方式

#### 对齐流程
1. 将Llama-Factory更新至v0.9.0版本（注意v0.9.0有requirements错误，accelerate版本需要3.34.0之后，否则会遇到问题）
2. 将step logs设成1，跑A800的前5步或者前10步，记录对应loss。
3. 启动910b，同样跑前5步或者前10步，第一步loss应该是一致的
4. loss曲线没有异常后，测试下游任务效果，如何和A800接近则说明对齐成功

#### 排查工具（遇到问题再使用）
##### py-spy
##### do-dump
##### Profiling


#### Faq

* 跑不起来
	检查库版本：主要是这几个库transformers、deepspeed、accelerate、torch、torch_npu这几个库版本有问题都有可能跑不起来
	检查进程：第一次跑的起来，后续跑不起来，往往是进程没杀干净（ctrl c是杀不干净的）。检查方式主要是看`npu-smi info`以及`ps -aux`查看是否还有进程。可以尝试使用`pkill -9 -f deepspeed`来杀掉所有进程

-  HCCL遇到超时报错
	表现形式：报错中提到timeout，或者tokenizer没跑完就结束
	解决方法：设置环境变量
	```bash
	export HCCL_CONNECT_TIMEOUT=7200
	export HCCL_EXEC_TIMEOUT=7200
	```

- 训练效率有问题
	检查机器网络配置：查看udp_port是否有配置打散
	```bash
	for i in $(seq 0 15); do echo "------ $i"; hccn_tool -i $i -udp -g;  done
	```
	检查是否为跨机房读取，io问题：将模型和数据复制到本机中（nas盘io较慢）
	检查启动方式是否正确：华为优化的启动很是是launcher启动
	检查Profiling：拿到Profiling后让架构华为同学帮忙排查

* 训练精度有问题
	排查方案：
	检查deepspeed版本：建议是0.14.3或者0.15.0版本，精度大概率是这个库带来的问题
	检查单机8卡：查看单机8卡的loss是否正常，如果不正常大概率是版本或者环境有问题
	检查同步流：如果单机或者较少机子loss正常，多机有问题大概率是同步流有问题。设置环境变量或者修改deepspeed配置
	检查数据流：使用do-dump工具采样具体数据，根据哪一步loss有问题采样多少步数据流。从后往前查看dump.json中e+数据，也就是inf数据（注意有一些inf是正常的，比如初始化算子）

* 第一步卡柱
	检查npu是否运行：`watch npu-smi info`若果`npu-smi info`显示npu利用率一直是0，大概率卡在数据处理中
	检查堆栈：使用py-spy来确认具体卡在哪个环节，如果已经到backward基本是正常，可能只是前几步比较慢会慢慢收敛（一般10-20step会收敛）
## Inference

910b推荐使用vllm来推理（才可以多卡推理）
多卡直接推理不行

#### Eval
910b支持opencomposs
