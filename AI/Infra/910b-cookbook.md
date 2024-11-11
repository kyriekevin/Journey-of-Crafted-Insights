---
tags:
  - llm
  - infra
  - 910b
date: 2024-11-10T21:00:00
---

# Merlin基础知识
## 集群
用户组：https://ml.bytedance.net/management/resource_management/group/966
CPU类型：arm
选择方式：需要选择HW ARM
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411110726747.png)
arm对应910b选择 HW ARM，不然集群显示卡数有问题

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

> [!NOTE]
> 查看CANN包等信息方法
>

Cann
```bash
cd /usr/local/Ascend/ascend-toolkit/latest/arm64-linux
cat ascend_toolkit_install.info
```
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411110710524.png)

driver
```bash
cd /usr/local/Ascend/driver
cat version.info
```

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411110713067.png)

# 910b基础知识
## Toolkit
类似`Nvidia GPU`，910b也有对应的`Npu/HCCL`等相关工具套件，在镜像中`cc`下，有`set_env.sh`脚本，如果环境变量有问题可以source

```bash
ls /usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 监控工具
类似于`nvidia-smi`，910b需要使用`npu-smi info`命令进行查看
AI Core指标对应Nvidia的SM（通常50-60%就算比较高）
HBM-Usage指标对应Nvidia的显存占用
![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411110724726.png)
主要需要查看的信息：AICore（NPU是否在运行）、HBM-Usage（显存占用）和Process id

> [!NOTE]
> npu利用率：采集的npu 拉上的 ai core单元的利用率 是低于mfu
> 总吞吐：根据模型size 数据量这些折算
>
> 如果是看相对速度，看token吞吐速度就可以了，模型硬件确定情况下跟mfu是正比关系，aicore利用率跟计算策略和实现有关系，不完全正比

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

> [!NOTE]
> `export ASCEND_RT_VISIBLE_DEVICES`来指定使用哪张卡

# 具体使用

## Train

### LLama-Factory

#### 基本信息
repo： https://code.byted.org/zhongyuzhe/llama-factory_npu
> [!NOTE]
> 如果选择基于已有llama-factory进行修改，请参考v0.9.0版本的是否有改动，并且accelerate版本是否正确。然后加上华为提供的patch算子修改包进行迁移

![image.png](https://raw.githubusercontent.com/zyzkyrie/img_auto/main/Obsidian/202411110800964.png)
以上为patch包内容，可以从代码仓库中进行复制（`install.sh`已经进行修改)

py3.8对应代码 https://code.byted.org/zhongyuzhe/llama-factory_npu/tree/90041e61c374b6794592b99715e76339f911e028
py3.9对应代码 https://code.byted.org/zhongyuzhe/llama-factory_npu/tree/a1230094e21c72d35be4aac8d127c8ddd460df82

> [!NOTE]
> py3.9对应的install.sh为华为提供，库版本是和华为对齐的，比较容易复现定位。py3.8为根据py3.9 install.sh改造适配py3.8可能有bug。

trail： https://ml.bytedance.net/development/instance/jobs/773ace128534c2df?tabState=task_config&trialId=36136963

> [!NOTE]
> py3.8和py3.9 trail区别只有镜像，同时注意启动代码应该是`$CODEBASE/src/llamafactory/launcher.py`而不是repo中的`$CODEBASE/src/train.py`

主要因为以下代码（`install.sh`)
```bash
cp $INSTALL_DIR/ascendcloud_patch/cli.py $INSTALL_DIR/LLaMA-Factory/src/llamafactory/
cp $INSTALL_DIR/ascendcloud_patch/launcher.py $INSTALL_DIR/LLaMA-Factory/src/llamafactory/
```


config：
```bash
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_RDMA_SL=4 # 不用设置
export HCCL_RDMA_TC=132 # 不用设置
export ASCEND_LAUNCH_BLOCKING=1 # 不用设置

# arm优化
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HOST_CACHE_CAPACITY=10
export COMBIND_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

| 环境变量                                       | 目的作用                                         |
| ------------------------------------------ | -------------------------------------------- |
| `HCCL_CONNECT_TIMEOUT`和`HCCL_EXEC_TIMEOUT` | 主要避免通信和执行超时                                  |
| `TASK_QUEUE_ENABLE`                        | 分发任务队列，会有一定额外显存开销                            |
| `CPU_AFFINITY_CONF`                        | 设置 CPU 亲和性配置，可以优化多核 CPU 的性能                  |
| `HOST_CACHE_CAPACITY`                      | 设置主机缓存容量                                     |
| `HCCL_RDMA_SL`和`HCCL_RDMA_TC`              | RDMA网络通信优化，理论上不用设置，但目前集群设置默认值有问题（多机多卡最好手动设置） |
| `ASCEND_LAUNCH_BLOCKING`                   | 控制同步流，影响训练速度。但可以避免内存踩踏、网络等一些问题               |
| `PYTORCH_NPU_ALLOC_CONF`                   | 控制是否开启虚拟内存                                   |

> [!NOTE]
> 在PyTorch2.1.0及以上版本中,使用torch_npu可以在一个进程中使用多个device卡,根据指定的device id将任务下发到期望的卡上执行。
> 开启虚拟内存特性时，不能使用单进程多卡特性
>
> deepspeed zero2 config中`overlap_comm`设置成False类似`ASCEND_LAUNCH_BLOCKING`效果

robust training：
1. 打开robust training开关
2. 在高级配置中配置一个多架构镜像
参考trail： https://ml.bytedance.net/development/instance/jobs/72b3f4da70ea0d65?tabState=task_config&trialId=35209173

> [!NOTE]
> executor的镜像就使用业务的镜像，不一定要用一样的；driver的镜像随便选择一个x86的python镜像就行

#### 使用方式

#### 对齐流程
1. 将Llama-Factory更新至v0.9.0版本（注意v0.9.0有requirements错误，accelerate版本需要3.34.0之后，否则会遇到问题）
2. 将step logs设成1，跑A800的前5步或者前10步，记录对应loss。
3. 启动910b，同样跑前5步或者前10步，第一步loss应该是一致的
4. loss曲线没有异常后，测试下游任务效果，如何和A800接近则说明对齐成功

> [!NOTE]
> llm一般是基于下游任务直接打分2个点以内算对齐

#### 排查工具（遇到问题再使用）
##### py-spy (检查python进程)
安装：`pip install py-spy`
使用：
1. 查看pid （`npu-smi info`)
2. `py-spy dump --pid xxx`

> [!NOTE]
> 一般用于检查长时间卡主的情况
> 如第一步卡主，查看是在数据处理还是已经`backward`

##### do-dump (检查数据流，精度相关)
工具网址: https://gitee.com/ascend/mstt
安装：`pip install mindstudio-probe`
使用：

设置环境变量
```bash
export DO_DUMP=TRUE
export ACC_CONFIG_PATH=xxx # 配置文件存放地址
```

设置配置文件
```json
{
    "task": "statistics",
    "dump_path": "", // 结果存放位置
    "rank": [], // 采集哪个rank
    "step": [], // 采集哪几个step，先定位哪个step开始出现问题
    "level": "L1",
    "enable_dataloader": false,

    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

正常启动运行（运行后dump文件会自动保存，建议梯度累积开小，否则dump很慢）

`dump.json`和`stack.json`

> [!NOTE]
> 让华为用工具检查
> 人眼看一般从后往前，查找e+看是否有异常的output
>
> 目前遇到的是narrow算子有异常值，和deepspeed同步流有关

##### Profiling
[昇腾Pytorch Profiling流程](https://bytedance.larkoffice.com/wiki/DwW1wGfKVit56hksAeGc5T2Kn8g)

#### Faq

* 跑不起来
	检查库版本：主要是这几个库transformers、deepspeed、accelerate、torch、torch_npu这几个库版本有问题都有可能跑不起来

	检查进程：第一次跑的起来，后续跑不起来，往往是进程没杀干净（ctrl c是杀不干净的）。检查方式主要是看`npu-smi info`以及`ps -aux`查看是否还有进程。可以尝试使用`pkill -9 -f deepspeed`来杀掉所有进程（报错提示信息为`DIST call hccl api failed`以及`Failed to initialize the HCCP process. Reason: Maybe the last training process is running.`)


-  HCCL遇到超时报错
	表现形式：报错中提到timeout，或者tokenizer没跑完就结束
	解决方法：设置环境变量
	```bash
	export HCCL_CONNECT_TIMEOUT=7200
	export HCCL_EXEC_TIMEOUT=7200
	```

- 训练效率有问题
	检查机器网络配置：查看udp_port是否有配置打散（多机多卡）
	```bash
	for i in $(seq 0 15); do echo "------ $i"; hccn_tool -i $i -udp -g;  done
	```
	检查通信环境变量：`HCCL_RDMA_SL`和`HCCL_RDMA_TC`（多机多卡）
	```bash
	cat /proc/xxx/environ | tr '\0' '\n' | grep hccl # xxx为npu-smi info查看pid
	```
	检查是否为跨机房读取，io问题：将模型和数据复制到本机中（nas盘io较慢）
	检查启动方式是否正确：华为优化的启动很是是launcher启动
	检查Profiling：拿到Profiling后让架构华为同学帮忙排查

* 训练精度有问题
	检查deepspeed版本：建议是0.14.3或者0.15.0版本，精度大概率是这个库带来的问题
	检查单机8卡：查看单机8卡的loss是否正常，如果不正常大概率是版本或者环境有问题
	检查同步流：如果单机或者较少机子loss正常，多机有问题大概率是同步流有问题。设置环境变量或者修改deepspeed配置
	检查数据流：使用do-dump工具采样具体数据，根据哪一步loss有问题采样多少步数据流。从后往前查看dump.json中e+数据，也就是inf数据（注意有一些inf是正常的，比如初始化算子）

* 第一步卡主
	检查npu是否运行：`watch npu-smi info`若果`npu-smi info`显示npu利用率一直是0，大概率卡在数据处理中
	检查堆栈：使用py-spy来确认具体卡在哪个环节，如果已经到backward基本是正常，可能只是前几步比较慢会慢慢收敛（一般10-20step会收敛）

- `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`
	版本问题：查看deepspeed、transformers、torch这三个主要库版本

- `AttributeError: 'DeepSpeedZeroOptimizer_Stage3' object has no attribute 'train'`
	版本问题：accelerate版本旧了，升级到0.34.0之后
## Inference

910b推荐使用vllm来推理（才可以多卡推理），直接推理推荐单进程单卡
多卡直接推理不行

> 使用`llama-factory cli chat`推理会报rope算子错误

#### Eval
910b支持opencomposs
参考文档： https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_90906.html
