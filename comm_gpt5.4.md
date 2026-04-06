# SGLang Ascend NPU 路径集合通信完全解析

## 导读：先补上集合通信最基础的概念

在正式进入 SGLang 的代码与时序之前，先把 **“集合通信是什么”** 讲清楚。否则后面看到：

- all-reduce
- all-gather
- reduce-scatter
- all-to-all
- TP / DP / EP / CP / PP

很容易只停留在“看懂了函数名”，却不知道它们为什么会出现在某个位置。

### 导读 1：什么是“集合通信”

集合通信（collective communication）可以理解成：

```text
一组 rank 按照某个共同约定，同时参加的一次多方通信操作
```

这里有 3 个最基本的关键词：

#### rank

rank 就是通信域里的一个参与者编号。

在大模型推理里，一个 rank 往往可以近似理解成：

- 一个 worker 进程
- 绑定一张 NPU 卡
- 在某个 process group 里有自己的编号

例如 8 张卡 TP=8 时，最常见就是：

```text
rank 0,1,2,3,4,5,6,7
```

#### world / process group

不是所有 rank 都必须每次都一起通信。

集合通信总是在一个 **通信域** 内发生。这个通信域在 PyTorch / SGLang 里通常体现为：

- `WORLD`
- `TP group`
- `ATTN_TP group`
- `MOE_EP group`

例如：

- `TP group=[0..7]` 表示 8 张卡一起做 TP 通信
- `ATTN_TP group=[0,1]` 表示只有这 2 张卡一起做某个 attention 小组通信

#### collective

collective 不是“一张卡给另一张卡发消息”的点对点通信，而是：

```text
多个 rank 一起参加同一个操作
```

例如：

- 大家一起求和：all-reduce
- 大家一起拼接：all-gather
- 大家一起边规约边切分：reduce-scatter

### 导读 2：为什么大模型推理一定离不开集合通信

因为单张卡放不下、算不完、也不一定跑得足够快，所以需要把模型或数据拆到多卡。

只要发生下面任意一种拆分，就几乎一定会引入集合通信：

#### Tensor Parallelism，TP

把一个层的权重按列或者按行切到多张卡上。

这会导致：

- 每张卡只算出一部分结果
- 后面必须通过 all-reduce 或 all-gather 把结果重新合起来

#### Data Parallelism，DP

把不同请求或不同 token shard 分给不同卡。

这会导致：

- 某些阶段每张卡只看到数据的一部分
- 进入下一阶段前，可能要做 gather / scatter / reduce-scatter

#### Expert Parallelism，EP

把 MoE 的 experts 分散到不同卡。

这会导致：

- token 要按路由结果被发到对应 expert 所在卡
- 常见表现是 all-to-all 或变体 dispatch / combine

### 导读 3：最常见的 5 个集合通信原语到底是什么意思

后文会反复出现这些原语，这里先用最朴素的话解释。

#### 1）All-Reduce

语义：

```text
每个 rank 都拿出一个张量
-> 按位做规约（通常是 SUM）
-> 每个 rank 都拿到同一个规约结果
```

最常见用途：

- TP 场景把 partial output 求和，还原成完整输出

直觉图：

```text
r0: A0
r1: A1
r2: A2
r3: A3

all-reduce(sum)

每个 rank 都得到 A0 + A1 + A2 + A3
```

#### 2）All-Gather

语义：

```text
每个 rank 都拿出自己的一段
-> 按某个维度拼接
-> 每个 rank 都拿到完整拼接结果
```

最常见用途：

- vocab parallel 的 logits 从 `[*, V/TP]` 拼回 `[*, V]`

直觉图：

```text
r0: [A]
r1: [B]
r2: [C]
r3: [D]

all-gather

每个 rank 都得到 [A, B, C, D]
```

#### 3）Reduce-Scatter

语义：

```text
先规约，再把结果切片分回各 rank
```

它可以看成：

```text
all-reduce + scatter
```

的融合版。

最常见用途：

- 既想完成规约，又不想让每张卡都保存完整结果
- DPA 或优化后的 TP 路径里很常见

#### 4）Broadcast

语义：

```text
一个 root rank 的数据，复制给组里所有 rank
```

常见用途：

- 广播小对象、配置、种子、元数据

#### 5）All-to-All

语义：

```text
每个 rank 给其他所有 rank 分别发送一段数据
同时也从其他所有 rank 接收一段数据
```

常见用途：

- EP>1 的 MoE token dispatch / combine

它的拓扑最复杂，也最容易成为瓶颈。

### 导读 4：为什么 all-reduce 和 all-gather 长得像，但本质不同

初学者最容易混淆的是这两个：

- all-reduce
- all-gather

你可以这样记：

#### all-reduce

```text
大家手里是“同形状、可规约”的 partial 值
目标是把它们合并成“同一个最终值”
```

#### all-gather

```text
大家手里是“不同切片”
目标是把切片重新拼起来
```

这正对应大模型里两类经典场景：

- row-parallel 输出汇总：all-reduce
- vocab shard logits 拼接：all-gather

### 导读 5：为什么推理里“通信位置”比“通信名字”更重要

同样是 all-reduce，放在不同位置，语义完全不同：

- embedding 后的 all-reduce：是在合并词表分片查表结果
- attention 后的 all-reduce：是在合并 `o_proj` 的 partial hidden
- MLP 后的 all-reduce：是在合并 `down_proj` 的 partial hidden
- MoE 后的 all-reduce：可能只是 TP 汇总，不一定是专家并行通信

所以本文后面的重点不是只列出“调用了哪些 collective”，而是：

```text
在 forward 的哪个阶段调用
+ 为什么此时必须通信
+ 通信前后的张量布局如何变化
```

***

## 导读：HCCL 是什么，它在 Ascend 上到底扮演什么角色

这一节基于公开的 HCCL 官方资料与开源仓库说明，先把 HCCL 的底层图景讲清楚，再回来看 SGLang 的代码路径就会轻松很多。

### 导读 6：HCCL 的定位

HCCL 全称是：

```text
Huawei Collective Communication Library
```

它是昇腾平台上的高性能集合通信库，职责和 CUDA 生态里的 NCCL 类似，但面向的是 Ascend NPU 软硬件栈。

从公开资料看，HCCL 的职责可以概括成一句话：

```text
在昇腾硬件和 CANN 软件栈上，为单机多卡、多机多卡提供高性能集合通信能力
```

这意味着在 PyTorch + torch\_npu + SGLang 这条链路里：

- SGLang 写的是高层模型逻辑
- `torch.distributed` 暴露统一分布式 API
- PyTorch 的 NPU backend 最终把 collective 落到 HCCL

也就是说，本文里看到的：

```python
dist.all_reduce(...)
dist.all_gather_into_tensor(...)
```

在 Ascend 场景下，底层通信执行者就是 HCCL。

### 导读 7：HCCL 在 CANN 里的分层位置

公开资料和开源说明都强调，HCCL 不是一个孤立库，而是 CANN 计算栈里的集合通信层。

可以把它粗略理解成：

```text
上层框架
-> torch.distributed / framework adapter
-> HCCL
-> Runtime / device scheduler / transport links
-> Ascend NPU
```

对本文最有用的理解是：

- **SGLang 决定什么时候通信**
- **HCCL 决定如何高效通信**

### 导读 8：HCCL 官方公开的三层结构

HCCL 开源仓库把自身概括成三层：

#### 通信框架

负责：

- 通信域管理
- 通信算子的业务串联
- 协同算法模块做选择
- 协同平台模块申请资源并下发任务

#### 通信算法

负责：

- 为具体 collective 选择和承载算法
- 根据通信域信息做任务编排
- 计算资源与执行计划

#### 通信平台

负责：

- 提供 NPU 之上与集合通信相关的资源抽象
- 提供维护、测试、维测能力

把它翻译成工程语言，可以理解为：

```text
框架层负责“组织”
算法层负责“怎么走”
平台层负责“跑在哪、拿什么资源跑”
```

### 导读 9：HCCL 支持哪些集合通信原语

公开资料明确提到 HCCL 支持：

- AllReduce
- Broadcast
- AllGather
- ReduceScatter
- AlltoAll

这和本文后面在 SGLang 里真正看到的几类通信正好一一对应：

- TP 汇总常用 AllReduce
- vocab / shard 拼接常用 AllGather
- DPA 回散常用 ReduceScatter
- EP token dispatch 常见 AlltoAll 或其变体

### 导读 10：HCCL 支持哪些底层链路

公开资料还提到 HCCL 可在以下高速链路上实现集合通信：

- HCCS
- RoCE
- PCIe

这意味着 HCCL 本身不是“某一种固定拓扑算法”，而是：

```text
在已有硬件链路之上，结合通信规模、拓扑和消息大小，选择合适算法和编排方式
```

对 SGLang 使用者来说，这一条的重要意义是：

- 你在 Python 代码里看到的都是 `all_reduce` / `all_gather`
- 但真正性能差异，很多时候来自 **底层链路质量与算法选择**

### 导读 11：HCCL 为什么会有多种算法

HCCL 开源仓库和官方资料都提到多种算法，例如：

- Mesh
- Ring
- Recursive Halving-Doubling / Halving-Doubling
- PairWise
- Star
- NHR
- NB
- AHC
- Pipeline

不需要把每一种都背下来，但要理解一个非常重要的原则：

```text
没有一种 collective 算法在所有场景下都最优
```

比如：

- Ring：关系简单，适合一些小规模或拥塞明显场景
- Halving-Doubling / RHD：步数更少，延迟通常更好，尤其适合 2 的幂规模
- Pipeline：适合大消息，能把 server 内和 server 间链路并发起来

这也解释了为什么同样是 8 卡 all-reduce：

- 小张量 decode
- 大张量 prefill

它们的最优通信行为未必相同。

### 导读 12：HCCL 的一个关键性能模型：α–β–γ

HCCL 开源说明中直接给出了常见的性能分析模型：

```text
D = α + nβ + nγ
```

其中：

- `α`：固定时延
- `β`：每 byte 传输时间
- `γ`：每 byte 规约计算时间
- `n`：数据量
- `p`：参与通信的 rank 数，影响通信步数

这个模型对理解本文后面的 prefill / decode 差异非常有帮助：

#### Prefill

- `n` 大
- 更容易被带宽项 `nβ` 支配

#### Decode

- 单次 `n` 小
- 更容易被固定时延 `α` 和步数支配

所以你后面会看到：

- 同样都是 collective
- prefill 更像“大块搬运”
- decode 更像“高频小包”

### 导读 13：HCCL 的执行并不是“Python 直接发网包”

官方资料明确提到，通信操作执行阶段，HCCL 会把：

- 通信算法编排
- 内存访问
- 通信任务

通过 Runtime 下发给昇腾设备的任务调度器，再由设备按编排执行。

这句话非常重要，它说明：

```text
HCCL 不只是一个 socket/网络发送库
而是 Runtime + 设备任务调度 + 通信算法编排协同工作的结果
```

因此，当你在 SGLang 里写：

```python
dist.all_reduce(x)
```

真正下面发生的是：

1. 框架把 collective 请求交给 HCCL
2. HCCL 结合通信域、消息大小、拓扑做算法选择和任务编排
3. Runtime 把任务下发到 Ascend 设备侧执行

### 导读 14：Ascend C 的 HCCL 高阶 API 进一步揭示了“设备侧任务”机制

公开资料中的 Ascend C HCCL 高阶 API 还揭示了一个更底层的视角：

- AI Core 侧可以调用 Prepare 接口准备通信任务
- 再通过 Commit 通知服务端执行
- 通过 Wait / Finalize 等机制等待或结束

这说明在更底层的 Ascend 编程模型里，HCCL 任务并不是“调用函数立刻同步做完”这么简单，而是：

```text
先描述通信任务
-> 再触发执行
-> 再等待完成
```

对本文需要掌握的程度来说，你不需要深入到 Ascend C API 级别，但应该形成这个印象：

> **HCCL 的底层实现是有明确任务编排、提交、调度过程的；Python 层的 collective 只是高层入口。**

### 导读 15：把 HCCL 放回到 SGLang 里，应该怎么理解

对本文最实用的理解可以浓缩成这 4 句话：

1. SGLang 决定 **哪里需要通信**
2. `torch.distributed` 决定 **用什么统一接口表达通信**
3. HCCL 决定 **在 Ascend 上如何高效执行这些 collective**
4. 最终性能同时受：
   - 通信原语类型
   - group 大小
   - tensor 大小
   - 底层链路
   - HCCL 算法选择
     共同影响

### 导读 16：这一节和后文的关系

如果你已经记住下面这张心智图，后面的 SGLang 代码就容易很多：

```text
SGLang 代码
-> 决定什么时候 all-reduce / all-gather / reduce-scatter
-> PyTorch NPU distributed backend
-> HCCL
-> HCCS / RoCE / PCIe 等链路
-> Ascend 设备执行
```

后面文档中的“通信步骤”分析，其实就是在回答：

```text
SGLang 为什么在这里把某个张量交给 HCCL 做某种 collective
```

***

## 0. 文档目标

本文以 **DeepSeekV3** 在 **SGLang Ascend NPU 路径**上的推理为主线，严格使用下面这个场景：

- TP = 8
- DP = 1
- EP = 1
- PP = 1
- PD 不分离（mixed，prefill / decode 不拆成两套服务）
- 设备后端 = Ascend NPU

本文只讨论 **SGLang 自己这条 PyTorch + torch\_npu + HCCL 的推理链路**，目标是把：

- 请求从进入服务到推理完成的全过程
- Ascend NPU 路径里真正发生的所有卡间集合通信
- 每次通信发生的位置、操作类型、通信内容、语义
- 为什么某些通信在这个配置里会发生，另一些不会发生
- 相关公共组件、公共封装、代码入口

一次性讲清楚。

***

## 1. 先给结论

如果你只想先抓住最核心的认识，可以先记住下面 8 点：

1. **DeepSeekV3 在 SGLang 里直接复用** **`DeepseekV2ForCausalLM`** **实现**，没有单独写一套前向。
2. 在 **TP=8, DP=1, EP=1, PP=1** 这个场景里，**真正参与集合通信的只有 TP 维度**。
3. Ascend 上默认分布式后端是 **HCCL**，不是 NCCL。
4. 本场景里 **不会发生 DP gather/scatter**，因为 DP=1。
5. 本场景里 **不会发生 EP all-to-all / token dispatch 跨卡通信**，因为 EP=1，且要保持 EP=1 时 `moe_a2a_backend` 必须是 `none`。
6. 本场景里 **不会发生 CP 通信**，因为 `attn_cp_size=1`。
7. DeepSeekV3 每层的注意力输出和 MLP/MoE 输出都使用 **TP all-reduce** 做结果汇总；最后 logits 使用 **TP all-gather** 拼回全词表。
8. 对一次正常 forward 来说，若模型有 `L` 层，则本场景下主干集合通信次数可以概括为：
   - 输入 embedding：1 次 all-reduce
   - 每层：2 次 all-reduce
   - 输出 logits：1 次 all-gather
   所以总计约等于：
   ```text
   1 + 2L + 1 = 2L + 2
   ```

也就是说，在这个配置里，**核心通信图景不是“MoE 一定 all-to-all”**，而是：

```text
Embedding all-reduce
-> 每层 Attention 输出 all-reduce
-> 每层 MLP / MoE 输出 all-reduce
-> Logits all-gather
```

***

## 2. 这篇文档的关键前提

### 2.1 DeepSeekV3 复用 DeepseekV2 实现

SGLang 里 DeepSeekV3 并没有单独重写模型 forward，而是直接继承 `DeepseekV2ForCausalLM`：

```python
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

这段代码来自：

- `python/sglang/srt/models/deepseek_v2.py`

所以本文后面所有关于 DeepSeekV3 的调用链，都会落到 `deepseek_v2.py`。

### 2.2 要让 EP 保持 1，MoE A2A backend 不能是 `ascend_fuseep`

这是本文最容易误解的点。

很多人一看到 “Ascend NPU + DeepSeekV3 + MoE”，就会自然联想到 “肯定会走 Ascend FuseEP 的 A2A 通信”。**但在你指定的这个场景里，答案是否定的。**

原因是：一旦 `moe_a2a_backend == "ascend_fuseep"`，SGLang 会直接把 `ep_size` 调整成 `tp_size`。

```python
if self.moe_a2a_backend == "ascend_fuseep":
    self.ep_size = self.tp_size
    logger.warning(
        f"Ascend fused EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
    )
```

这段代码来自：

- `python/sglang/srt/server_args.py`

所以：

- 如果你坚持 **EP=1**
- 又要分析 **TP=8, DP=1, EP=1** 的真实运行路径

那么本文默认的 MoE 配置就必须是：

```text
moe_a2a_backend = none
```

也正因为如此，本文主线里 **不会出现 DeepEP / FuseEP / AllToAll 的跨卡专家分发**。

***

## 3. 这个场景下，8 张卡是怎么分组的

### 3.1 分布式初始化入口

SGLang 在 `ModelRunner` 初始化时完成分布式环境和模型并行组初始化：

```python
init_distributed_environment(
    backend=backend,
    world_size=self.tp_size * self.pp_size,
    rank=self.tp_size * self.pp_rank + self.tp_rank,
    local_rank=self.gpu_id,
    distributed_init_method=dist_init_method,
    timeout=self.server_args.dist_timeout,
    moe_a2a_backend=self.server_args.moe_a2a_backend,
)
initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    attention_data_parallel_size=self.dp_size,
    pipeline_model_parallel_size=self.pp_size,
    expert_model_parallel_size=self.moe_ep_size,
    attention_context_model_parallel_size=self.attn_cp_size,
    moe_data_model_parallel_size=self.moe_dp_size,
    duplicate_tp_group=self.server_args.enable_pdmux,
)
initialize_dp_attention(
    server_args=self.server_args,
    model_config=self.model_config,
)
if is_npu():
    register_sgl_tp_rank(self.gpu_id)
```

这段代码来自：

- `python/sglang/srt/model_executor/model_runner.py`

### 3.2 Ascend 默认 backend 是 HCCL

`parallel_state.py` 明确把 NPU 映射到 HCCL：

```python
_DEVICE_TO_DISTRIBUTED_BACKEND = {
    "cuda": "nccl",
    "xpu": "xccl",
    "hpu": "hccl",
    "cpu": "gloo",
    "npu": "hccl",
    "musa": "mccl",
}


def get_default_distributed_backend(device: str) -> str:
    return _DEVICE_TO_DISTRIBUTED_BACKEND.get(device, "gloo")
```

### 3.3 HCCL WORLD 组初始化

```python
pg_options = get_torch_distributed_pg_options()

torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=rank,
    timeout=timeout,
    pg_options=pg_options,
)
```

对 Ascend 而言，这里的 `backend` 就是 `hccl`。

### 3.4 本场景下的具体分组结果

假设全局 rank 为：

```text
rank 0, 1, 2, 3, 4, 5, 6, 7
```

并且：

- TP = 8
- DP = 1
- EP = 1
- PP = 1
- attn\_cp\_size = 1

那么 `initialize_model_parallel()` 算出来的组关系是：

#### WORLD 组

```text
[0, 1, 2, 3, 4, 5, 6, 7]
```

#### TP 组

```text
[0, 1, 2, 3, 4, 5, 6, 7]
```

#### Attention TP 组

因为：

```text
attn_tp_size = tp_size / attn_cp_size / attn_dp_size
             = 8 / 1 / 1
             = 8
```

所以：

```text
ATTN_TP = TP = [0, 1, 2, 3, 4, 5, 6, 7]
```

#### Attention CP 组

因为 `attn_cp_size = 1`，所以每个 CP 组都是单卡：

```text
[0], [1], [2], [3], [4], [5], [6], [7]
```

#### MoE EP 组

因为 `EP = 1`，每个专家并行组都是单卡：

```text
[0], [1], [2], [3], [4], [5], [6], [7]
```

#### MoE DP 组

因为 `moe_dp_size = 1`，这里也是单卡组：

```text
[0], [1], [2], [3], [4], [5], [6], [7]
```

#### MoE TP 组

因为：

```text
moe_tp_size = tp_size / moe_ep_size / moe_dp_size
            = 8 / 1 / 1
            = 8
```

所以：

```text
MOE_TP = TP = [0, 1, 2, 3, 4, 5, 6, 7]
```

#### PP 组

因为 `PP = 1`，每个流水段都是单卡：

```text
[0], [1], [2], [3], [4], [5], [6], [7]
```

***

## 4. 先理解：哪些集合通信在本文场景里不会发生

这一节非常重要，因为它决定了你读代码时应该把注意力放在哪里。

### 4.1 不会发生 DP collectives

`initialize_dp_attention()` 里，如果 `dp_size == 1`，那么 Attention DP 大小就被设成 1：

```python
if enable_dp_attention:
    _ATTN_DP_SIZE = dp_size
    ...
else:
    _ATTN_DP_SIZE = 1
    _LOCAL_ATTN_DP_SIZE = 1
```

因此下面这些通信都不会进入真实执行路径：

- `dp_gather_partial`
- `dp_gather_replicate`
- `dp_scatter`
- `dp_reduce_scatter_tensor`

它们是给 **DP>1** 场景准备的。

### 4.2 不会发生 EP A2A collectives

`create_moe_dispatcher()` 在 `moe_a2a_backend.is_none()` 时，选择的是 `StandardDispatcher`：

```python
def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    a2a_backend = get_moe_a2a_backend()
    if a2a_backend.is_none():
        return StandardDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_deepep()
        or a2a_backend.is_mooncake()
        or a2a_backend.is_mori()
        or a2a_backend.is_nixl()
    ):
        return MaybeTboDeepEPDispatcher(...)
    elif a2a_backend.is_ascend_fuseep():
        return NpuFuseEPDispatcher(...)
```

而 `StandardDispatcher` 里，只有某些 FP4 特殊路径才会触发 `all_gatherv / reduce_scatterv`；普通 `EP=1 + moe_a2a_backend=none` 主线不会走这些路径：

```python
if should_use_flashinfer_cutlass_moe_fp4_allgather():
    topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
        [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
    )
...
if should_use_flashinfer_cutlass_moe_fp4_allgather():
    hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
    get_tp_group().reduce_scatterv(
        global_hidden_states,
        output=hidden_states,
        sizes=get_dp_global_num_tokens(),
    )
```

这意味着：

- **没有 expert token 的跨卡 all-to-all**
- **没有 DeepEP/FuseEP 的 dispatch/combine 跨卡交换**
- **没有 EP 维度上的集合通信**

### 4.3 不会发生 CP collectives

因为 `attn_cp_size = 1`，所以下面这些 CP 通信也都不会实际触发：

- `attn_cp_all_gather_into_tensor`
- `attn_cp_reduce_scatter_tensor`
- `cp_all_gather_rerange_output`

### 4.4 不会发生 PP 点对点传递

因为 `PP = 1`，所以不会有多流水段之间的 hidden state 传输。

### 4.5 不会发生 PD 分离下的 KV 传输

你给定的是 **PD mixed**，不是 prefill / decode 分离部署，因此：

- 不会有 prefill worker 到 decode worker 的 KV 传输
- 不会有 disaggregation transfer backend 的跨角色数据搬运

换句话说，**请求进入后，真正相关的卡间通信基本全部集中在 TP 组上。**

***

## 5. 请求从发送到推理完成的主链路

为了理解“通信发生在什么时候”，先把整个调用链拉直。

### 5.1 服务入口到 Scheduler

高层主链路是：

```text
HTTP /generate
-> TokenizerManager.generate_request
-> Scheduler.handle_generate_request
-> Scheduler 主循环组 batch
-> Scheduler.run_batch
-> TpModelWorker.forward_batch_generation
-> ModelRunner.forward
-> DeepseekV3(实际是 DeepseekV2) forward
```

这里真正开始触发 NPU 模型计算，是在：

- `TpModelWorker.forward_batch_generation`
- `ModelRunner.forward`

之前的 HTTP、tokenize、scheduler、batch 组织，**都不是 HCCL 集合通信阶段**。

### 5.2 Worker 进入模型 forward

`TpModelWorker.forward_batch_generation()`：

```python
if model_worker_batch is not None:
    forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

if self.pp_group.is_last_rank:
    out = self.model_runner.forward(
        forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
        skip_attn_backend_init=skip_attn_backend_init,
    )
```

`ModelRunner.forward()` 再根据模式分流：

```python
if forward_batch.forward_mode.is_decode():
    ret = self.forward_decode(...)
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
    ret, can_run_graph = self.forward_extend(...)
```

所以：

- **Prefill** 走 `forward_extend`
- **Decode** 走 `forward_decode`

本文后面会看到：**它们的集合通信种类几乎相同，只是 tensor 大小和调用频次不同。**

***

## 6. Ascend NPU 集合通信的公共组件

在进入具体的每一步通信前，先认识公共通信封装。

### 6.1 HCCL ProcessGroup 选项

Ascend 路径下，SGLang 会为 WORLD 组和 MoE 相关组创建 HCCL 选项：

```python
def get_torch_distributed_pg_options(group_name=None):
    if not _is_npu:
        return None

    if group_name is not None and "moe" not in group_name:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE") or os.environ.get("HCCL_BUFFSIZE") or 200
    )
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options
```

它的意义是：

- 真正底层通信库是 **HCCL**
- HCCL 缓冲区大小可以通过环境变量调优

### 6.2 GroupCoordinator：SGLang 的统一通信门面

`GroupCoordinator` 是 SGLang 最关键的通信抽象层。每个逻辑 group 都会有：

- `device_group`
- `cpu_group`

NPU 场景里还会挂上 `NpuCommunicator`：

```python
self.npu_communicator: Optional[NpuCommunicator] = None
if use_npu_communicator and self.world_size > 1:
    self.npu_communicator = NpuCommunicator(group=self.device_group)
```

### 6.3 NpuCommunicator：Ascend 设备侧的最薄封装

```python
class NpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_npu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ...
        dist.all_gather_into_tensor(output_tensor, x, group=self.group)
        ...
        return output_tensor
```

也就是说，在 Ascend 上，很多最终集合通信都会落成：

- `torch.distributed.all_reduce(..., backend=hccl)`
- `torch.distributed.all_gather_into_tensor(..., backend=hccl)`

### 6.4 上层统一 API：`communication_op.py`

模型代码不会到处直接写 `dist.all_reduce`，而是统一调这些接口：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return get_tp_group().all_gather(input_, dim)
```

这使得模型层代码和具体 backend 解耦。

***

## 7. 为什么本场景下每层基本只有两类主通信

DeepSeekV3 的每个 decoder layer 在这个配置下，模式基本可以归纳成：

```text
输入 hidden
-> prepare_attn
-> attention
-> prepare_mlp
-> MLP 或 MoE
-> postprocess_layer
```

其中，决定通信模式的核心是 `LayerScatterModes`：

```python
class LayerScatterModes:
    ...
    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return (
                ScatterMode.SCATTERED
                if (
                    not get_moe_a2a_backend().is_none()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                )
                else ScatterMode.FULL
            )
        else:
            return (
                ScatterMode.SCATTERED
                if enable_moe_dense_fully_dp()
                else ScatterMode.FULL
            )
```

在本文设定下：

- `get_moe_a2a_backend().is_none() == True`
- `enable_moe_dense_fully_dp() == False`
- `DP=1`
- `CP=1`

所以不管当前层是：

- 稠密层
- 稀疏 MoE 层

其 `mlp_mode` 都是 `FULL`，`layer_output_mode` 也都会回到 `TP_ATTN_FULL`。

这带来的直接结果是：

- **不会走 scattered / gather / reduce-scatter 主线**
- **每层主要就是两个 TP all-reduce**

第一个发生在：

- Attention 输出汇总到 MLP 输入前

第二个发生在：

- MLP 或 MoE 输出汇总回完整 hidden state 时

***

## 8. 请求进入后，真正发生的集合通信总时序

下面开始进入本文最重要的部分：**以一次真实请求为主线，按时间顺序梳理所有卡间集合通信。**

为了便于描述，定义两个符号：

- `T_prefill`：这次 prefill batch 的总 token 数
- `B_decode`：这次 decode batch 的活跃请求数（因为每个请求本轮只解一个 token）
- `H`：hidden size
- `V`：vocab size
- `L`：decoder 层数

### 8.1 请求进入服务到进入模型前：没有 HCCL 集合通信

从：

```text
HTTP 请求
-> tokenizer
-> scheduler
-> batch 组织
-> 构造 ForwardBatch
```

这一段主要是：

- Python 控制流
- 队列调度
- 张量准备
- KV slot 分配

不会触发 NPU 间的 HCCL collective。

所以真正的通信起点，是 **模型 forward 的第一层输入 embedding**。

***

## 9. 第 1 类通信：输入 embedding 之后的 TP all-reduce

### 9.1 位置

代码在：

- `python/sglang/srt/layers/vocab_parallel_embedding.py`

### 9.2 代码

```python
def forward(self, input_):
    if self.tp_size > 1:
        masked_input, input_mask = get_masked_input_and_mask(...)
    else:
        masked_input = input_

    with use_symmetric_memory(
        get_tp_group(), disabled=not is_allocation_symmetric()
    ):
        output_parallel = self.quant_method.embedding(self, masked_input.long())

    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        if not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                output_parallel = attn_tp_all_reduce(output_parallel)
            else:
                output_parallel = tensor_model_parallel_all_reduce(output_parallel)
    return output_parallel
```

### 9.3 操作类型

```text
All-Reduce(SUM) over TP group [0..7]
```

### 9.4 通信内容

每张卡本地持有的是 **词表的一部分 shard**，因此它只能查到：

- 自己负责的 vocab slice 对应的 embedding
- 不属于自己分片的 token，会被 mask 成 0

于是每张卡得到的是一个局部结果：

```text
[T, H]
```

然后通过 all-reduce 求和，把 8 张卡的局部 embedding 拼成完整结果。

### 9.5 这次通信的语义

这不是在“做统计归约”，而是在“**把分片词表查表结果重新还原成完整 embedding**”。

可以把它理解成：

- rank0 只认识部分词
- rank1 只认识另一部分词
- ...
- 每张卡先给出自己知道的 embedding，不知道的填 0
- 最后用 SUM 合并，得到完整 embedding

### 9.6 Prefill 和 Decode 的区别

- **Prefill**：通信张量大小约为 `[T_prefill, H]`
- **Decode**：通信张量大小约为 `[B_decode, H]`

通信类型相同，只是 prefill 一次处理的 token 更多。

***

## 10. 每个 Decoder Layer 的整体形态

单层主 forward 代码是：

```python
hidden_states, residual = self.layer_communicator.prepare_attn(
    hidden_states,
    residual,
    forward_batch,
    quant_format,
)

hidden_states = self.self_attn(
    positions=positions,
    hidden_states=hidden_states,
    forward_batch=forward_batch,
    zero_allocator=zero_allocator,
    llama_4_scaling=llama_4_scaling,
    layer_scatter_modes=self.layer_scatter_modes,
)

hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)

should_allreduce_fusion = (
    self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
        forward_batch
    )
)

use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
    forward_batch
)

hidden_states = self.mlp(
    hidden_states,
    forward_batch,
    should_allreduce_fusion,
    use_reduce_scatter,
    gemm_output_zero_allocator,
)

if not should_allreduce_fusion:
    hidden_states, residual = self.layer_communicator.postprocess_layer(
        hidden_states, residual, forward_batch
    )
```

这段代码来自：

- `python/sglang/srt/models/deepseek_v2.py`

这段代码特别重要，因为它告诉我们：

- Attention 前后、MLP/MoE 前后，通信并不是“随便散落”的
- 它们被 `LayerCommunicator` 组织成固定的阶段

***

## 11. 每层第 1 次主通信：Attention 输出汇总到 MLP 输入前的 TP all-reduce

### 11.1 先看一个容易误解的点：Attention 里的 `o_proj` 本身并不直接做 all-reduce

DeepSeekV3 的注意力输出投影 `o_proj` 是 `RowParallelLinear`，但构造时：

```python
self.o_proj = RowParallelLinear(
    self.num_heads * self.v_head_dim,
    self.hidden_size,
    bias=False,
    quant_config=quant_config,
    reduce_results=reduce_results,
    prefix=add_prefix("o_proj", prefix),
    tp_rank=attn_tp_rank,
    tp_size=attn_tp_size,
)
```

而在 decoder layer 初始化里，传给 attention 的是：

```python
self.self_attn = DeepseekV2AttentionMLA(
    ...
    reduce_results=False,
    ...
)
```

所以 `o_proj` 的行为是：

- 只产出 **局部 partial output**
- 暂时不做 all-reduce

这正是后面 `prepare_mlp()` 需要补一发 all-reduce 的原因。

### 11.2 NPU Attention 核心计算本身不做卡间 collective

NPU MLA 核心路径：

```python
def forward_mla_core_npu(...):
    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    attn_output = attn_output.contiguous()
    torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_bmm_output)

    return output
```

这里看到的都是：

- 本地 Q/K/V 计算
- 本地 NPU attention kernel
- 本地 `o_proj`

**没有 HCCL collective。**

### 11.3 真正的 all-reduce 发生在 `prepare_mlp()`

`prepare_mlp()` 最终会进入 `CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual()`：

```python
if not handled:
    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    if _is_npu and context.cache is not None:
        _ = prepare_weight_cache(hidden_states, context.cache)
    hidden_states, residual = layernorm(hidden_states, residual)
```

### 11.4 操作类型

```text
All-Reduce(SUM) over TP group [0..7]
```

### 11.5 通信内容

这次 all-reduce 传输的是 **attention 输出投影后的 partial hidden states**，形状大致是：

```text
[T, H]
```

其中：

- prefill 时 `T = T_prefill`
- decode 时 `T = B_decode`

### 11.6 这次通信的语义

由于 `o_proj` 是 row-parallel：

- 每张卡只算出了输出 hidden state 的一部分“加和项”
- 要进入下一步 MLP / MoE，必须先恢复成完整 hidden state

所以这里的 all-reduce 本质上是在做：

```text
把 Attention 输出从 8 份 partial sum 合并成完整 hidden
```

### 11.7 为什么它在这里而不是 `o_proj` 里面发生

因为 SGLang 把“通信 + layernorm + 后续调度”集中到了 `LayerCommunicator` 里处理，带来的好处是：

- 便于统一优化不同层的 scatter/gather 模式
- 便于以后接 DP / CP / speculative / overlap
- 便于把通信和下游 layernorm 组合考虑

对理解代码的人来说，这一点非常关键：

> **看 DeepSeekV3 的 TP 通信，不能只盯着线性层本身；很多通信被延后到** **`LayerCommunicator`** **里做。**

***

## 12. 每层第 2 次主通信（稠密层）：MLP 输出的 TP all-reduce

### 12.1 代码位置

MLP 定义：

```python
class DeepseekV2MLP(nn.Module):
    ...
    self.gate_up_proj = MergedColumnParallelLinear(...)
    self.down_proj = RowParallelLinear(
        intermediate_size,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=reduce_results,
        prefix=add_prefix("down_proj", prefix),
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
```

forward：

```python
gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x, _ = self.down_proj(
    x,
    skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
)
return x
```

`RowParallelLinear.forward()`：

```python
if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
    if self.use_dp_attention_reduce:
        output = get_attention_tp_group().all_reduce(output_parallel)
    else:
        output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

### 12.2 操作类型

```text
All-Reduce(SUM) over TP group [0..7]
```

### 12.3 通信内容

传输的是 `down_proj` 之后的 MLP 输出 partial hidden states，形状仍然是：

```text
[T, H]
```

### 12.4 这次通信的语义

`down_proj` 是 row-parallel，因此每张卡只算出完整输出的一部分求和项。要得到真正的 MLP 输出，必须做 all-reduce：

```text
完整 MLP 输出 = 各 rank 的 down_proj partial output 之和
```

### 12.5 在本文场景里它为什么稳定存在

因为：

- NPU 路径不走 CUDA 的 PyNccl / custom AR 分支
- DP=1，不会改走 DP reduce-scatter
- 本场景也没有需要触发的特殊 fusion 路径

所以这次 all-reduce 是最典型、最稳定的一类 TP 通信。

***

## 13. 每层第 2 次主通信（稀疏层）：MoE 输出的 TP all-reduce

DeepSeekV3 不是所有层都是普通 MLP，后面很多层会变成 MoE 层。这里最容易出现误区：

> “MoE 层一定会有 all-to-all”

在本文场景中，这句话是错的。

### 13.1 为什么本文场景的 MoE 不走 all-to-all

`DeepseekV2MoE` 在 `moe_a2a_backend=none` 时不会走 `forward_deepep()`，而是走普通路径：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
    gemm_output_zero_allocator: BumpAllocator = None,
) -> torch.Tensor:
    if not self._enable_a2a_moe:
        ...
        return self.forward_normal(...)
    else:
        return self.forward_deepep(hidden_states, forward_batch)
```

而 `_enable_a2a_moe` 只有在 deepep / mooncake / nixl / mori / ascend\_fuseep / flashinfer 等后端下才会变成 True。

### 13.2 普通 MoE 路径的最终通信

`forward_normal()` 的关键尾部逻辑是：

```python
final_hidden_states = self.experts(
    hidden_states,
    topk_output,
)
...
if (
    self.tp_size > 1
    and not should_allreduce_fusion
    and not use_reduce_scatter
    and not should_use_flashinfer_cutlass_moe_fp4_allgather()
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
return final_hidden_states
```

而 `FusedMoE.forward_impl()` 末尾也有同样的语义：

```python
if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

### 13.3 操作类型

```text
All-Reduce(SUM) over TP group [0..7]
```

### 13.4 通信内容

传输的是 **MoE 层最终输出的 partial hidden states**，形状也是：

```text
[T, H]
```

### 13.5 这次通信的语义

虽然这是一层 MoE，但在 **EP=1** 下：

- 不做跨卡专家分发
- token 不会在 expert 维度上被发送到别的卡
- routed experts 计算完之后，仍然只需要在 **TP 维度** 上把 partial output 求和

所以它本质上仍然是：

```text
Row-parallel / TP-sharded 输出的合并
```

而不是：

```text
MoE expert all-to-all
```

### 13.6 对初学者最重要的理解

在 SGLang 里，**“模型是 MoE”** 和 **“运行时一定有 EP 集合通信”** 不是同一件事。

只有当：

- `EP > 1`
- 并且启用了相应 A2A backend

时，MoE 才会进入真正的跨卡专家通信路径。

本文这个场景下，MoE 只是“模型结构是稀疏层”，但 **运行时通信仍然主要是 TP all-reduce**。

***

## 14. `postprocess_layer()` 在本文场景里为什么没有额外集合通信

很多读代码的人会继续追：

> `postprocess_layer()` 还会不会再做 gather / scatter / reduce-scatter？

在本文场景里，基本不会。

`postprocess_layer()` 最终会走 `CommunicateSummableTensorPairFn`。而当输入输出模式 group size 相同的时候，会退化成 trivial：

```python
if context.is_same_group_size(
    hidden_states_input_mode, output_mode
) and context.is_same_group_size(residual_input_mode, output_mode):
    return CommunicateSummableTensorPairFn._trivial
```

也就是说，在：

- DP=1
- CP=1
- EP=1
- `mlp_mode=FULL`
- `layer_output_mode=TP_ATTN_FULL`

这个组合下，`postprocess_layer()` 不会再引入新的集合通信。

所以单层的主通信数目仍然保持为 **2 次 all-reduce**。

***

## 15. 最后一类主通信：Logits 的 TP all-gather

### 15.1 位置

代码在：

- `python/sglang/srt/layers/logits_processor.py`

### 15.2 关键代码

```python
hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
    hidden_states, logits_metadata
)

logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

if self.do_tensor_parallel_all_gather:
    if self.use_attn_tp_group:
        logits = self._gather_attn_tp_logits(logits)
    else:
        logits = tensor_model_parallel_all_gather(logits)
```

而 `LogitsProcessor.__init__()` 在本文场景里会走普通 TP gather 路线：

```python
self.use_attn_tp_group = get_global_server_args().enable_dp_lm_head
...
else:
    self.do_tensor_parallel_all_gather = (
        not skip_all_gather and get_tensor_model_parallel_world_size() > 1
    )
    self.do_tensor_parallel_all_gather_dp_attn = (
        self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1
    )
```

由于 `DP=1`，所以这里不会有 DP gather/scatter，最终就是普通：

```python
logits = tensor_model_parallel_all_gather(logits)
```

### 15.3 操作类型

```text
All-Gather over TP group [0..7]
```

### 15.4 通信内容

这里每张卡持有的是词表的一部分 shard 对应的 logits：

```text
local logits:  [T_or_B, V/8]
global logits: [T_or_B, V]
```

于是需要 all-gather 把 8 片 vocab logits 拼回完整词表。

### 15.5 这次通信的语义

它和 embedding 阶段刚好相反：

- 输入 embedding 阶段：词表分片查表后用 **all-reduce** 合成完整 hidden
- 输出 logits 阶段：词表分片 logits 用 **all-gather** 拼回完整 vocab

这非常适合初学者记忆：

```text
Embedding 端：按 vocab 分片 -> 结果求和
Logits 端：按 vocab 分片 -> 结果拼接
```

### 15.6 Prefill 和 Decode 的区别

- **Prefill**：若要返回某些位置的 logits / logprob，传输大小接近 `[T_prefill, V/8]`
- **Decode**：通常只关心当前步的 next-token logits，传输大小接近 `[B_decode, V/8]`

同样，**通信类型不变，主要是数据规模和频次不同。**

***

## 16. 采样阶段没有新的 HCCL 集合通信

`model_runner.sample()` 做的是：

- logits 后处理
- top-k / top-p / 温度等采样
- 生成 next token id

这一阶段不再引入新的卡间 collective。

所以一次 forward 的通信终点，通常就停在：

```text
logits all-gather
```

***

## 17. Prefill 和 Decode：通信种类相同，但粒度完全不同

这是理解线上性能的关键。

### 17.1 Prefill

Prefill 特征：

- 一次处理整个 prompt
- token 数多
- attention 计算重
- 每次 collective 的 tensor 更大

在本文场景下，prefill 的主通信仍然是：

1. embedding all-reduce
2. 每层 attention 输出 all-reduce
3. 每层 MLP / MoE 输出 all-reduce
4. logits all-gather

只是这里的 `T` 是整个 batch 的 prompt token 总量。

### 17.2 Decode

Decode 特征：

- 每个请求每轮只生成 1 个 token
- 单轮 tensor 更小
- 但调用轮数很多
- 更容易被通信延迟支配

在本文场景下，decode 仍然是同一套 collective，只是：

```text
T_prefill 变成 B_decode
```

也就是：

- 每次 all-reduce 更小
- 但每生成一个 token 都要重复一遍整层通信链

### 17.3 一个很实用的性能直觉

对 TP=8 的 DeepSeekV3 来说：

- **Prefill 更容易被大张量计算和大张量通信共同影响**
- **Decode 更容易被“频繁小 all-reduce / all-gather 的时延”影响**

所以如果线上 decode TPS 不理想，很多时候问题不在 attention kernel 本身，而在：

- TP all-reduce 延迟
- 最终 logits all-gather 延迟
- HCCL 拓扑、链路、buffer 配置

***

## 18. 这个场景下的一次 forward，可以压缩成什么时序图

### 18.1 Prefill 时序图

```text
请求进入
-> Scheduler 组 prefill batch
-> TpModelWorker.forward_batch_generation
-> ModelRunner.forward_extend
-> VocabParallelEmbedding
   -> TP all-reduce
-> Layer 0
   -> NPU attention 本地计算
   -> TP all-reduce（attention 输出汇总）
   -> MLP/MoE 本地计算
   -> TP all-reduce（MLP/MoE 输出汇总）
-> Layer 1
   -> 同上
...
-> Layer L-1
   -> 同上
-> LM Head / LogitsProcessor
   -> TP all-gather（拼完整 vocab logits）
-> Sampling / 输出结果
```

### 18.2 Decode 时序图

```text
请求进入 decode 循环某一步
-> Scheduler 组 decode batch
-> TpModelWorker.forward_batch_generation
-> ModelRunner.forward_decode
-> VocabParallelEmbedding
   -> TP all-reduce
-> Layer 0
   -> NPU attention 本地计算（读 KV cache）
   -> TP all-reduce
   -> MLP/MoE 本地计算
   -> TP all-reduce
...
-> Layer L-1
   -> 同上
-> LM Head / LogitsProcessor
   -> TP all-gather
-> Sampling
-> 生成本轮 token
```

你会发现：**时序结构完全相同，只有 token 规模不同。**

***

## 19. 一张总表：本文场景下所有真实发生的集合通信

| 序号 | 阶段              | 代码位置                                                     | 通信组           | 操作         | 通信内容                                | 语义                              |
| -- | --------------- | -------------------------------------------------------- | ------------- | ---------- | ----------------------------------- | ------------------------------- |
| 1  | 输入 embedding    | `VocabParallelEmbedding.forward`                         | TP = `[0..7]` | all-reduce | `[T, H]` 的局部 embedding 输出           | 合并 vocab 分片查表结果                 |
| 2  | 每层 Attention 后  | `LayerCommunicator.prepare_mlp`                          | TP = `[0..7]` | all-reduce | `[T, H]` 的 attention partial output | 合并 `o_proj` 的 row-parallel 局部输出 |
| 3  | 每层 Dense MLP 后  | `RowParallelLinear.forward`                              | TP = `[0..7]` | all-reduce | `[T, H]` 的 MLP partial output       | 合并 `down_proj` 的局部输出            |
| 4  | 每层 Sparse MoE 后 | `DeepseekV2MoE.forward_normal` / `FusedMoE.forward_impl` | TP = `[0..7]` | all-reduce | `[T, H]` 的 MoE partial output       | 合并 TP 维度上的 MoE 输出               |
| 5  | 最终 logits       | `LogitsProcessor._get_logits`                            | TP = `[0..7]` | all-gather | `[T_or_B, V/8]` 的局部 logits shard    | 拼回完整词表 logits                   |

说明：

- 表中第 3 和第 4 是二选一，因为某一层不是 Dense MLP 就是 Sparse MoE。
- 所以对任意一层来说，主通信数目基本稳定为 **2**：
  - attention 后 1 次 all-reduce
  - MLP/MoE 后 1 次 all-reduce

***

## 20. 一张总表：本文场景下不会触发的集合通信

| 通信                                                                 | 代码位置                                                        | 为什么不触发                          |
| ------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------- |
| `dp_gather_partial` / `dp_gather_replicate`                        | `dp_attention.py` / `logits_processor.py`                   | `DP=1`                          |
| `dp_scatter` / `dp_reduce_scatter_tensor`                          | `dp_attention.py`                                           | `DP=1`                          |
| `attn_tp_reduce_scatter_tensor`                                    | `communicator.py`                                           | 本文配置下层模式不进入 scattered 路径        |
| `attn_cp_all_gather_into_tensor` / `attn_cp_reduce_scatter_tensor` | `dp_attention.py`                                           | `CP=1`                          |
| DeepEP / FuseEP dispatch/combine                                   | `token_dispatcher/deepep.py` / `token_dispatcher/fuseep.py` | `EP=1` 且 `moe_a2a_backend=none` |
| MoE all-to-all                                                     | 各类 A2A dispatcher                                           | 本文不是 EP 并行场景                    |
| PP send / recv                                                     | pipeline 相关路径                                               | `PP=1`                          |
| PD 分离 KV 传输                                                        | disaggregation 路径                                           | 本文是 PD mixed                    |

***

## 21. “MoE 明明很复杂，为什么本文场景还是只有 TP 通信？”

这是本文最值得反复强调的一点。

### 21.1 MoE 的“模型结构复杂”不等于“通信拓扑复杂”

MoE 带来的复杂性主要有两种：

1. **计算复杂性**
   - router
   - top-k experts
   - routed experts
   - shared experts
2. **通信复杂性**
   - token dispatch
   - token combine
   - expert all-to-all
   - expert parallel 负载均衡

本文场景里，真正发生的是第 1 类，**不是第 2 类**。

### 21.2 因为 EP=1，专家并没有跨卡拆分

当 `EP=1` 时，SGLang 不需要把 token 发到“别的专家卡”上去执行，因此不会触发：

- token dispatch 跨卡发送
- token combine 跨卡回收
- expert all-to-all

此时 MoE 只是在本卡做：

- gate
- topk
- local experts compute

最后再在 TP 维度把 partial hidden 做 all-reduce。

这就是为什么：

> **DeepSeekV3 是 MoE 模型，但在 TP=8, DP=1, EP=1 的 Ascend 路径下，通信主角仍然是 TP all-reduce，而不是 EP all-to-all。**

***

## 22. 这套通信和 Ascend NPU 后端的关系到底在哪里

很多人读完上面的路径会问：

> 这些 all-reduce / all-gather 看起来和 CUDA 时代很像，那 Ascend 的特殊性到底体现在哪？

答案是：**集合通信的语义相似，但底层后端和 attention kernel 完全不同。**

### 22.1 集合通信后端换成了 HCCL

SGLang 的 TP all-reduce / all-gather 语义和 CUDA 路径类似，但 Ascend 上实际落的是：

```text
torch.distributed + HCCL ProcessGroup
```

而不是：

```text
PyNccl / NCCL custom all-reduce
```

### 22.2 Attention 计算和 KV cache 读写走的是 NPU 专用实现

例如 NPU MLA/MHA 路径会进入：

- `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

这些路径会使用：

- `torch_npu`
- NPU fused attention kernel
- NPU paged attention / KV cache 接口

也就是说：

- **通信层**：还是 SGLang 的 TP 抽象，但 backend 是 HCCL
- **计算层**：是真正的 Ascend NPU 专用 attention / KV cache 实现

### 22.3 这也是为什么本文要把“本地 NPU 计算”和“卡间集合通信”严格区分

例如在 attention 内部：

- 写 KV cache
- 读 KV cache
- 执行 fused infer attention

这些都很关键，但它们不是“集合通信”。

它们和 all-reduce / all-gather 共同构成整条推理链，但职责不同：

- **NPU 算子**负责本卡算得快
- **HCCL collective**负责 8 卡之间把 TP 分片重新合并

***

## 23. 如果把一次请求拆成“通信视角”的逐步清单

假设一个请求经历：

1. 一次 prefill
2. 后面连续 `M` 轮 decode

那么它的主干集合通信次数大致是：

### 23.1 Prefill 阶段

```text
1 次 embedding all-reduce
+ 每层 2 次 all-reduce
+ 1 次 logits all-gather
```

即：

```text
2L + 2
```

### 23.2 每一轮 decode 阶段

仍然是：

```text
2L + 2
```

### 23.3 整个请求从发送到结束

如果 decode 生成了 `M` 个 token，那么总主干 collective 数可粗略写成：

```text
(2L + 2) * (1 + M)
```

这个公式非常粗，但对理解“为什么 decode 通信压力大”特别有帮助。

因为即使单次 decode 张量更小，它也要 **重复整层通信链很多次**。

***

## 24. 需要重点掌握的公共代码文件

下面这组文件，基本覆盖了本文所有通信分析的“骨架”。

### 24.1 分布式与进程组

- `python/sglang/srt/distributed/parallel_state.py`

职责：

- 选择 backend（NPU -> HCCL）
- 初始化 WORLD / TP / ATTN\_TP / MOE\_\* / PP 组
- 封装 `GroupCoordinator`
- 提供 all-reduce / all-gather / reduce-scatter 统一实现

### 24.2 NPU 通信器

- `python/sglang/srt/distributed/device_communicators/npu_communicator.py`

职责：

- NPU 上的 `all_reduce`
- NPU 上的 `all_gather`

### 24.3 TP 上层 API

- `python/sglang/srt/distributed/communication_op.py`

职责：

- 给模型层提供统一的 `tensor_model_parallel_all_reduce`
- 给模型层提供统一的 `tensor_model_parallel_all_gather`

### 24.4 层内通信编排

- `python/sglang/srt/layers/communicator.py`

职责：

- 管理 attention / MLP 前后的通信阶段
- 决定什么时候 all-reduce，什么时候 gather/scatter
- 统一 LayerNorm 与通信顺序

### 24.5 DP / Attention TP / CP 辅助通信

- `python/sglang/srt/layers/dp_attention.py`

职责：

- 定义 DP gather/scatter
- 定义 `attn_tp_all_gather_into_tensor`
- 定义 `attn_tp_reduce_scatter_tensor`

虽然本文很多函数不会实际触发，但它是理解“为什么没触发”的关键背景文件。

### 24.6 线性层并行

- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`

职责：

- 定义 row-parallel / column-parallel 线性层
- 定义 embedding 与 LM head 的分片与汇总方式

### 24.7 Logits 汇总

- `python/sglang/srt/layers/logits_processor.py`

职责：

- 最终 logits 的 TP all-gather
- DP attention 场景下的 gather/scatter

### 24.8 DeepSeekV3 主模型

- `python/sglang/srt/models/deepseek_v2.py`

职责：

- DeepSeekV3 的真实 forward 主体
- Decoder layer 组织
- Dense MLP / Sparse MoE 逻辑

### 24.9 Ascend NPU Attention 实现

- `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

职责：

- NPU MLA/MHA 前向
- KV cache 读写
- NPU 专用 attention kernel 调用

### 24.10 MoE 公共后端选择

- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/standard.py`

职责：

- 选择 MoE dispatcher
- 区分 `none / deepep / ascend_fuseep / nixl / ...`
- 决定有没有 A2A 通信

***

## 25. 如果未来把配置改掉，通信图景会怎么变

为了让这篇文档更“可迁移”，这里补一个对照。

### 25.1 如果 DP > 1

会新增：

- `dp_gather_partial`
- `dp_gather_replicate`
- `dp_scatter`
- `dp_reduce_scatter_tensor`

此时通信就不再只是 TP all-reduce / all-gather。

### 25.2 如果 EP > 1

会新增：

- token dispatch
- token combine
- expert parallel collectives

尤其是当：

- `moe_a2a_backend=deepep`
- `moe_a2a_backend=ascend_fuseep`

时，MoE 会进入完全不同的跨卡通信形态。

### 25.3 如果启用 `ascend_fuseep`

SGLang 会把：

```text
ep_size <- tp_size
```

这时就不再是本文的 `EP=1` 场景了，MoE 会进入真正的 Ascend FuseEP 路径。

### 25.4 如果 PP > 1

会新增：

- pipeline stage 之间的 hidden state 传递

### 25.5 如果 PD 分离

会新增：

- prefill / decode 角色之间的 KV 传输

***

## 26. 最终总结

把整篇文档浓缩成一句话：

> 在 SGLang 的 Ascend NPU 路径里，以 **DeepSeekV3 + TP=8 + DP=1 + EP=1 + PD mixed** 为例，真正发生的卡间集合通信几乎全部是 **TP 组上的 HCCL all-reduce 和 all-gather**；DeepSeekV3 虽然是 MoE 模型，但由于 **EP=1 且** **`moe_a2a_backend=none`**，并不会进入跨卡专家分发的 A2A 通信路径。

如果再展开成 5 句话，就是：

1. **Ascend backend = HCCL**，分组和集合通信统一由 `parallel_state.py + GroupCoordinator + NpuCommunicator` 管理。
2. **请求进入服务到进入模型前**，没有 HCCL collective；真正的通信从模型 forward 开始。
3. **输入 embedding** 先做 1 次 TP all-reduce，把 vocab-shard 查表结果合成完整 hidden。
4. **每个 decoder layer** 固定有 2 次主通信：attention 输出 1 次 TP all-reduce，MLP 或 MoE 输出再 1 次 TP all-reduce。
5. **最终 logits** 做 1 次 TP all-gather，把 `[*, V/8]` 拼回 `[*, V]`，之后采样结束。

所以，如果你从“集合通信视角”去看这条路径，最应该记住的不是 “DeepSeekV3 是 MoE”，而是：

```text
本文配置下的通信主旋律 = TP all-reduce + logits all-gather
```

这就是 SGLang Ascend NPU 路径在这个部署形态下最核心、最稳定、也最值得掌握的通信事实。

***

## 27. 追加说明：`dp_size` 在 SGLang 里有两种完全不同的含义

在继续看你新要求的两个场景前，必须先把一个非常容易混淆的点讲透：

```text
SGLang 里的 dp_size，并不总是“单个 distributed world 里真的存在一个 DP 维度”
```

它有两种语义：

### 27.1 语义 A：普通 DP 副本

当 **没有开启** **`enable_dp_attention`** 时，`dp_size` 表示：

- 启动多少套独立的 TP 副本
- controller 把请求路由到某一套副本
- 副本之间彼此独立，不参加同一个 forward 的 collective

这一点最关键的证据在 `DataParallelController.launch_dp_schedulers()`：

```python
for dp_rank in range(server_args.dp_size):
    ...
    thread = threading.Thread(
        target=self.launch_tensor_parallel_group_thread,
        args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
    )
    ...
    base_gpu_id += (
        server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
    )
```

这段逻辑的语义非常直白：

- 每个 `dp_rank` 启动一套自己的 TP worker
- 下一套副本的 `base_gpu_id` 直接跳过一整组 `tp_size * pp_size`

这说明它是在 **复制副本**，不是在一个 world 内切出新的 DP 维度。

### 27.2 语义 B：DP Attention

只有当 **开启** **`enable_dp_attention`** 时，`dp_size` 才会进入：

- attention 维度重排
- `attn_dp_size`
- `attn_tp_size`
- DP gather/scatter

这一点在 `ModelRunner.__init__()` 里写得非常直接：

```python
self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
```

也就是说：

- 不开 DPA：模型内部看到的 `dp_size` 就是 1
- 开 DPA：模型内部才真的把 `dp_size` 当成 attention 维度的一部分

### 27.3 因此本文后面两节的真实含义是

你要求新增的两个场景，源码语义上应分别理解为：

1. **`TP=4, DP=2`**
   - 默认理解为：**不开** **`enable_dp_attention`**
   - 即：2 套独立副本，每套副本内部是 TP=4
2. **`TP=8, DP=4 且开启 DPAttention`**
   - 明确理解为：**一个 8 卡 world**
   - attention 被切成：
     - `attn_dp=4`
     - `attn_tp=2`

也正因为这个差异，这两个场景虽然都写着 `dp_size > 1`，但它们的集合通信形态完全不同。

***

## 28. 场景二：TP=4，DP=2（默认未开启 DPAttention）

这一节分析的是：

- TP = 4
- DP = 2
- EP = 1
- PP = 1
- **未开启** **`enable_dp_attention`**
- PD mixed
- Ascend NPU

为了便于理解，下面假设机器上有 8 张卡：

```text
GPU 0 1 2 3 4 5 6 7
```

在这个场景下，它们不是一个 8 卡大 world，而是两套互不通信的 4 卡副本：

```text
副本 0: [0,1,2,3]
副本 1: [4,5,6,7]
```

### 28.1 启动方式：controller 启动两套独立 TP 副本

高层入口仍然是 `DataParallelController`，但它走的是普通 DP 分支：

```python
def launch_dp_schedulers(self, server_args, port_args):
    base_gpu_id = 0

    for dp_rank in range(server_args.dp_size):
        ...
        thread = threading.Thread(
            target=self.launch_tensor_parallel_group_thread,
            args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
        )
        ...
        base_gpu_id += (
            server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
        )
```

这段代码来自：

- `python/sglang/srt/managers/data_parallel_controller.py`

它说明：

- `dp_rank=0` 启动第一套 TP=4 副本
- `dp_rank=1` 启动第二套 TP=4 副本
- 两套副本的 GPU 集合完全分开

### 28.2 请求如何进入某个副本

在普通 DP 模式下，请求会被 controller 负载均衡到某一个副本，例如 round-robin：

```python
def round_robin_scheduler(self, req: Req):
    ...
    if self.status[self.round_robin_counter]:
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(
            self.workers
        )
```

这意味着：

- 单个请求只会进入 **一个** DP 副本
- 这个请求的整个 prefill / decode 生命周期，都在该副本内部完成
- **副本 0 和副本 1 之间没有这个请求的卡间集合通信**

### 28.3 模型内部实际上看不到 `dp_size=2`

`ModelRunner` 初始化时明确写了：

```python
self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
```

所以在这个场景下：

```text
server_args.dp_size = 2
enable_dp_attention = False
=> model_runner.dp_size = 1
```

这件事的后果非常大：

- `initialize_model_parallel()` 里的 `attention_data_parallel_size` 实际上传入的是 1
- `initialize_dp_attention()` 里 `_ATTN_DP_SIZE` 也会是 1
- 模型层内部完全不会走 DP gather/scatter 路径

### 28.4 每套副本内部的真实分组

对任意一个副本来说，它的 distributed world 就是 4 张卡。

因此副本内部的分组关系是：

#### WORLD 组

```text
[0,1,2,3]    或    [4,5,6,7]
```

#### TP 组

```text
[0,1,2,3]    或    [4,5,6,7]
```

#### Attention TP 组

因为模型内部 `attn_dp_size=1`，所以：

```text
attn_tp_size = tp_size / attn_cp_size / attn_dp_size
             = 4 / 1 / 1
             = 4
```

因此：

```text
ATTN_TP = TP
```

#### Attention DP 组

不存在真实的 DPA 语义，可以直接理解为：

```text
attn_dp_size = 1
```

#### EP / CP / PP

仍然与前一版文档中的 `TP=8, DP=1, EP=1` 场景一样：

- EP=1，不会有专家跨卡分发
- CP=1，不会有 CP collectives
- PP=1，不会有 pipeline 传递

### 28.5 这条链路的本质

这个场景最重要的理解是：

> `TP=4, DP=2` 并不是“一个请求同时动用 8 张卡，其中 4 张做 TP、2 张做 DP”。

它真正的含义是：

> 同时维护 **两套 4 卡 TP 推理副本**，每个请求只会进入其中一套。

所以从“单请求集合通信”的视角看：

- 这个请求只会在 **4 张卡内部** 做 collective
- 另一套 4 张卡完全不参与这个请求

### 28.6 请求从发送到完成的主流程

如果把一次请求从发送到结束按时序展开，它是：

```text
HTTP 请求
-> TokenizerManager
-> DataParallelController
-> 被路由到某个 DP 副本
-> 该副本内部的 Scheduler 组 batch
-> TpModelWorker.forward_batch_generation
-> ModelRunner.forward
-> DeepSeekV3(实际复用 DeepseekV2) forward
-> 输出结果
```

与前一版文档相比，区别只有一个：

```text
前一版：系统里只有 1 套 TP=8 副本
本节：系统里有 2 套 TP=4 副本，单请求只进入其中一套
```

### 28.7 真实发生的集合通信

一旦请求进入某个 4 卡副本，它内部的通信模式就和前一版文档完全同构，只是 group size 从 8 变成 4。

#### 28.7.1 输入 embedding：4 卡 TP all-reduce

embedding 代码仍然是：

```python
if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    if not get_attn_tp_context().input_scattered:
        if self.use_attn_tp_group:
            output_parallel = attn_tp_all_reduce(output_parallel)
        else:
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
```

此时：

- `self.tp_size = 4`
- `use_attn_tp_group = False`

所以真实操作是：

```text
All-Reduce(SUM) over TP group of size 4
```

语义仍然是：

- 每张卡只持有 1/4 词表 shard
- 先做本地 embedding 查表
- 再 all-reduce 求和，恢复完整 hidden states

#### 28.7.2 每层 attention 输出：4 卡 TP all-reduce

`prepare_mlp()` 里的核心逻辑不变：

```python
if not handled:
    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    ...
    hidden_states, residual = layernorm(hidden_states, residual)
```

因此每层 attention 之后都要做一次：

```text
All-Reduce(SUM) over TP group of size 4
```

语义是：

- attention 的 `o_proj` 是 row-parallel
- 每张卡只有局部 partial output
- 需要在 4 张卡内求和，恢复完整 hidden

#### 28.7.3 每层 MLP 输出：4 卡 TP all-reduce

Dense MLP 的 `down_proj` 路径仍然是：

```python
if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
    ...
    output = tensor_model_parallel_all_reduce(output_parallel)
```

所以 dense layer 的第二次主通信是：

```text
All-Reduce(SUM) over TP group of size 4
```

#### 28.7.4 每层 MoE 输出：4 卡 TP all-reduce

因为本节依然是：

- EP=1
- `moe_a2a_backend=none`

所以 MoE 不会进入 EP all-to-all，只会在 TP 维度汇总输出：

```python
if (
    self.tp_size > 1
    and not should_allreduce_fusion
    and not use_reduce_scatter
    and not should_use_flashinfer_cutlass_moe_fp4_allgather()
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

因此 sparse layer 的第二次主通信也是：

```text
All-Reduce(SUM) over TP group of size 4
```

#### 28.7.5 最终 logits：4 卡 TP all-gather

logits 末尾仍然走：

```python
if self.do_tensor_parallel_all_gather:
    if self.use_attn_tp_group:
        logits = self._gather_attn_tp_logits(logits)
    else:
        logits = tensor_model_parallel_all_gather(logits)
```

由于这里没有启用 DPA，所以：

- `use_attn_tp_group = False`
- `get_tensor_model_parallel_world_size() = 4`

因此最终是：

```text
All-Gather over TP group of size 4
```

### 28.8 本场景下不会发生的通信

这一节尤其重要，因为很多人会误以为 `DP=2` 一定会出现 DP collectives。

事实上，对单请求来说，这里不会发生：

- 副本 0 和副本 1 之间的 all-reduce
- `dp_gather_partial`
- `dp_gather_replicate`
- `dp_scatter`
- `dp_reduce_scatter_tensor`
- `attn_tp_reduce_scatter_tensor`
- EP all-to-all

原因分别是：

1. **副本间不共享一个 world**
2. **模型内部看到的** **`dp_size=1`**
3. **EP=1**

### 28.9 用一句话总结本节

`TP=4, DP=2` 在 SGLang 里，默认应理解为：

```text
两套互相独立的 TP=4 Ascend 推理副本
```

因此，对单请求而言，它的集合通信主线仍然是：

```text
4 卡 embedding all-reduce
-> 每层 4 卡 attention 输出 all-reduce
-> 每层 4 卡 MLP/MoE 输出 all-reduce
-> 4 卡 logits all-gather
```

和前一版 `TP=8, DP=1` 的差异，不是“多了 DP collectives”，而是：

```text
group size 从 8 变成 4，系统里额外多了一套独立副本
```

***

## 29. 场景三：TP=8，DP=4，且开启 DPAttention

这一节分析的是：

- TP = 8
- DP = 4
- `enable_dp_attention = True`
- EP = 1
- PP = 1
- PD mixed
- Ascend NPU

这一节与上一节最本质的区别是：

```text
上一节：2 套独立 TP=4 副本
本节：1 个 8 卡 world，在 world 内重排出 attn_dp=4、attn_tp=2
```

也就是说，本节终于进入了 **真正的 DPAttention 集合通信路径**。

### 29.1 启动方式已经完全变了：不再是多副本，而是一个统一 world

controller 这时不再走 `launch_dp_schedulers()`，而是走：

```python
def launch_dp_attention_schedulers(
    self, server_args: ServerArgs, port_args: PortArgs
):
    ...
    self.launch_tensor_parallel_group(
        server_args, port_args, 0, None, broadcasted_ports
    )
```

这说明：

- 不再按 `dp_rank` 启动 4 套副本
- 而是启动 **一整套 TP=8 的统一 worker group**

### 29.2 DPA 下 `dp_rank` 是从 `tp_rank` 现算出来的

在 `launch_tensor_parallel_group()` 里，DPA 分支会这样计算：

```python
if server_args.enable_dp_attention:
    _, _, dp_rank = compute_dp_attention_world_info(
        server_args.enable_dp_attention,
        tp_rank,
        server_args.tp_size,
        server_args.dp_size,
        server_args.attn_cp_size,
    )
    rank_port_args = PortArgs.init_new(
        server_args, dp_rank, worker_ports
    )
    rank_port_args.nccl_port = port_args.nccl_port
```

这段代码的含义是：

- 8 个 rank 仍然共享同一个 distributed world
- 但 controller 额外给它们打上了逻辑 `dp_rank`
- 所有这些 rank 仍然使用 **同一个 HCCL world / 同一个通信端口**

也就是说：

```text
这里的 DP 不是“复制副本”
而是“同一 world 内的 attention 维度重分组”
```

### 29.3 `attn_tp_size` 和 `attn_dp_rank` 是怎么推出来的

`dp_attention.py` 中的核心公式是：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size

if not enable_dp_attention:
    attn_dp_rank = 0
else:
    attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

代入本节参数：

- `tp_size = 8`
- `dp_size = 4`
- `attn_cp_size = 1`

得到：

```text
attn_dp_size = 4
attn_tp_size = 8 / 4 / 1 = 2
attn_tp_rank = tp_rank % 2
attn_dp_rank = tp_rank // 2
```

### 29.4 8 个 rank 的真实映射

于是 8 个 rank 的映射关系就是：

| 全局 rank | attn\_dp\_rank | attn\_tp\_rank |
| ------- | -------------- | -------------- |
| 0       | 0              | 0              |
| 1       | 0              | 1              |
| 2       | 1              | 0              |
| 3       | 1              | 1              |
| 4       | 2              | 0              |
| 5       | 2              | 1              |
| 6       | 3              | 0              |
| 7       | 3              | 1              |

这张表是理解整个 DPA 路径最关键的一张表。

### 29.5 显式创建的组和逻辑上的“列”

`initialize_model_parallel()` 显式创建的 attention TP 组是：

```text
[0,1]
[2,3]
[4,5]
[6,7]
```

因为代码是：

```python
for tp_group_idx in range(num_tensor_model_parallel_groups):
    for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
        st = (
            tp_group_idx * tensor_model_parallel_size
            + cp_dp_combined_idx * attn_tp_size
        )
        en = (
            tp_group_idx * tensor_model_parallel_size
            + (cp_dp_combined_idx + 1) * attn_tp_size
        )
        ranks = list(range(st, en))
        group_ranks.append(ranks)
```

但注意：

> 源码里**没有单独创建一个** **`ATTN_DP`** **process group**。

所谓的 DP lane 是逻辑上推导出来的两列：

```text
lane 0: [0,2,4,6]   # attn_tp_rank = 0
lane 1: [1,3,5,7]   # attn_tp_rank = 1
```

后面很多 DP collectives，本质上就是：

- 用整个 `TP group`
- 再配合 `ATTN_TP group`
- 组合出逻辑上的 DP 聚合效果

### 29.6 `ScatterMode` 在 DPA 下的直觉图

`communicator.py` 里的 `ScatterMode` 文档非常值得看：

```python
"""
Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
Model input/output: [ab, ab, cd, cd] for four ranks respectively
SCATTERED: [a, b, c, d]
TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
FULL: [abcd, abcd, abcd, abcd]
"""
```

把它推广到本节的 `TP=8, DP=4, attn_tp=2`，可以这样理解：

- `SCATTERED`：每个逻辑 DP shard 只保留自己那一份 token
- `TP_ATTN_FULL`：每个 2 卡 `ATTN_TP` 小组内，两张卡持有同一份该 DP shard 的完整 attention 输入
- `FULL`：所有 8 张卡都看到全局所有 token

这三个模式之间的切换，就是本节集合通信的核心。

### 29.7 本节最重要的结论：通信不再只是“TP all-reduce”

前一版文档中的 `TP=8, DP=1` 场景，每层主线是：

```text
attention 后 1 次 TP all-reduce
+ MLP/MoE 后 1 次 TP all-reduce
```

而本节 DPA 场景中，attention 前后的通信语义变成了：

```text
attention shard -> DP gather -> FULL
FULL -> MLP / MoE
MLP / MoE 输出 -> DP scatter / DP reduce-scatter -> attention shard
```

所以这时的通信主角已经变成：

- `dp_gather_partial`
- `dp_gather_replicate`
- `dp_reduce_scatter_tensor`
- `attn_tp_all_gather_into_tensor`
- `attn_tp_reduce_scatter_tensor`

同时，底层仍然会复用：

- TP group 上的 all-reduce / all-gather / reduce-scatter
- ATTN\_TP group 上的 all-gather / reduce-scatter

***

## 30. 场景三的请求主流程

### 30.1 请求进入系统

高层仍然是：

```text
HTTP 请求
-> TokenizerManager
-> DataParallelController
-> Scheduler
-> TpModelWorker.forward_batch_generation
-> ModelRunner.forward
```

但与普通 DP 最大的区别在于：

- 请求虽然会被路由到某个逻辑 `dp_rank`
- 但模型执行时，8 张卡仍然是同一个 HCCL world
- 每层需要在这个 world 内做 DPA 协调

### 30.2 `_forward_raw()` 会走 DPA 专用 batch 预处理

`ModelRunner._forward_raw()` 里：

```python
if forward_batch.global_num_tokens_cpu is not None:
    forward_batch.prepare_mlp_sync_batch(self)
else:
    forward_batch.prepare_attn_tp_scatter_input(self)
```

在 DPA 场景里，关键就是 `prepare_mlp_sync_batch()`。

它会：

1. 对各个 DP shard 的 token 数做对齐
2. 选择 `DpPaddingMode`
3. 分配 `global_dp_buffer`
4. 为后面的 gather/scatter 做元数据准备

### 30.3 DPA 为什么要区分 `SUM_LEN` 和 `MAX_LEN`

`prepare_mlp_sync_batch()` 里有这段逻辑：

```python
dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
    self.is_extend_in_batch, global_num_tokens
)
self.dp_padding_mode = dp_padding_mode

if dp_padding_mode.is_max_len():
    max_num_tokens = max(global_num_tokens)
    global_num_tokens = [max_num_tokens] * sync_group_size
    buffer_len = max_num_tokens * sync_group_size
else:
    buffer_len = sum(global_num_tokens)
```

它的语义是：

- **SUM\_LEN**
  - 不把所有 DP rank 补到相同长度
  - 总 buffer 长度是各 rank token 数之和
  - 更省内存
- **MAX\_LEN**
  - 把所有 rank 补到相同长度
  - 通信更规整
  - 更利于某些 gather / reduce-scatter 路径

### 30.4 一个非常实用的直觉

在本节场景里：

- **Prefill** 通常更偏向 `SUM_LEN`
- **Decode** 更可能落到 `MAX_LEN`

原因是：

- prefill 各 DP rank 的 token 数往往差异大
- decode 时每个 rank 常常只对应相近数量的活跃请求

这会直接影响后面到底走哪条 collective 组合路径。

***

## 31. 场景三的第 1 类通信：Embedding 不再是 8 卡 TP all-reduce，而是 2 卡 `ATTN_TP` all-reduce

这是 DPA 场景最容易被忽略、但最重要的变化之一。

### 31.1 模型初始化时 embedding 的并行组已经换了

`DeepseekV2Model` 创建 embedding 时写的是：

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    use_attn_tp_group=is_dp_attention_enabled(),
)
```

而 `VocabParallelEmbedding.__init__()` 中：

```python
if self.enable_tp:
    if use_attn_tp_group:
        tp_rank = get_attention_tp_rank()
        self.tp_size = get_attention_tp_size()
    else:
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
```

这意味着在本节场景里：

```text
embedding 的分片维度不再是 full TP=8
而是 attention TP=2
```

### 31.2 因此真实的 embedding 通信组是 4 个 2 卡小组

也就是：

```text
[0,1]
[2,3]
[4,5]
[6,7]
```

每个小组内部做自己的 embedding shard 合并。

### 31.3 为什么要这样设计

原因是：

- 模型输入 / 输出在 DPA 下的自然形态是 `TP_ATTN_FULL`
- attention 真正工作的最内层小组是 `ATTN_TP`
- 如果 embedding 直接按 8 卡 full TP 分片，后面还得立刻重排

所以 SGLang 在 DPA 场景里，干脆让 embedding 一开始就对齐到 `ATTN_TP`。

### 31.4 真实操作

仍然是 `VocabParallelEmbedding.forward()`：

```python
if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    if not get_attn_tp_context().input_scattered:
        if self.use_attn_tp_group:
            output_parallel = attn_tp_all_reduce(output_parallel)
        else:
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
```

但本节里：

- `self.use_attn_tp_group = True`
- `self.tp_size = 2`

因此真实发生的是：

```text
All-Reduce(SUM) over each ATTN_TP group of size 2
```

### 31.5 通信内容与语义

每个 2 卡小组：

- 共享同一个逻辑 `attn_dp_rank`
- 各自持有 1/2 vocab shard
- 先查本地 embedding
- 再在 2 卡小组内求和，恢复该 DP shard 对应的输入 hidden states

所以 embedding 阶段不再是全 8 卡通信，而是：

```text
4 个并行发生的 2 卡 all-reduce
```

***

## 32. 场景三的每层第 1 次主通信：Attention 输出之后的 DP gather

这一节是 DPA 路径的核心。

### 32.1 旧路径为什么不成立了

在 `DP=1` 的普通 TP 路径里，attention 输出后直接做：

```text
TP all-reduce -> 完整 hidden -> 进入 MLP
```

但本节里 attention 输出还是按 `TP_ATTN_FULL` / DP shard 组织的，不能直接进入 full TP=8 的 MLP。

因此 `prepare_mlp()` 必须先把各个 DP shard 的 token 聚成 FULL。

### 32.2 代码位置

`prepare_mlp()` 最终在 DPA 情况下会进入：

```python
if context.attn_dp_size != 1:
    ...
    hidden_states, local_hidden_states = (
        get_global_dp_buffer(),
        hidden_states,
    )
    dp_gather_partial(hidden_states, local_hidden_states, forward_batch)

    if not use_layer_norm_before_gather:
        dp_scatter(residual, hidden_states, forward_batch)
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)
```

注意这里的关键词已经变成了：

```text
dp_gather_partial
```

而不是：

```text
tensor_model_parallel_all_reduce
```

### 32.3 `dp_gather_partial` 的语义

它的目标是：

```text
把 4 个 DP shard 上的 token，收集成 FULL 视图
```

但它不是简单建一个 `ATTN_DP` 组去 all-gather，因为源码没有这个组。

它是通过两种不同实现完成的：

- `_dp_gather_via_all_reduce`
- `_dp_gather_via_all_gather`

### 32.4 `SUM_LEN` 路径：底层用 full TP all-reduce 实现 DP gather

当 `dp_padding_mode` 是 `SUM_LEN` 时，`dp_gather_partial()` 会走：

```python
global_tokens.fill_(0)
...
memcpy_triton(
    global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
)
...
global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
```

这条路径的语义是：

1. 每个 rank 先把自己本地那段 token 拷到全局 buffer 的对应位置
2. 其他位置补 0
3. 然后在 **整个 TP=8 group** 上做 all-reduce

因为各 rank 的有效数据放在不重叠的区间里，所以 all-reduce 之后的效果等价于：

```text
把 4 个 DP shard 拼接起来
```

这是一种非常巧妙的实现：

- 语义上是在做 DP gather
- 底层实现上却是 **8 卡 TP all-reduce**

### 32.5 `MAX_LEN` 路径：底层是 `ATTN_TP reduce-scatter + TP all-gather`

当 `dp_padding_mode` 是 `MAX_LEN` 时，会走：

```python
if get_attention_tp_size() == 1:
    get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
    return

...
scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
    get_attention_tp_rank()
]
get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)
```

在本节里 `attn_tp_size=2`，所以这条路径真实会拆成两步：

#### 第一步

```text
Reduce-Scatter over each ATTN_TP group of size 2
```

组分别是：

```text
[0,1] [2,3] [4,5] [6,7]
```

#### 第二步

```text
All-Gather over full TP group of size 8
```

### 32.6 这次通信的语义

无论是 `SUM_LEN` 还是 `MAX_LEN`，它们的高层语义都是一样的：

```text
把 attention 之后按 DP shard 切开的 token 重新聚成 FULL token 视图
```

这样 dense MLP / sparse MoE 才能在 full TP=8 的逻辑下继续计算。

### 32.7 这次通信与前一版文档最大的差异

前一版 `TP=8, DP=1` 里，attention 后是：

```text
1 次 TP all-reduce
```

而本节里 attention 后变成：

- 语义上：`DP gather`
- 实现上：
  - `SUM_LEN`：1 次 full TP all-reduce
  - `MAX_LEN`：1 次 2 卡 reduce-scatter + 1 次 8 卡 all-gather

这就是 DPA 下集合通信复杂度显著上升的根源之一。

***

## 33. 场景三的每层第 2 次主通信：MLP / MoE 之后的回散

完成 `dp_gather_partial()` 后，hidden states 已经变成 FULL 视图，于是可以进入：

- Dense MLP
- 或 Sparse MoE

这时后半段通信是否还是简单 all-reduce，取决于 padding mode。

### 33.1 `SUM_LEN` 路径：MLP / MoE 先做 full TP all-reduce，再本地 `dp_scatter`

如果不是 `MAX_LEN` 优化路径，那么 dense MLP 的 `down_proj` 仍然会做普通：

```python
if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

MoE 也类似：

```python
final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

随后 `postprocess_layer()` 里的 `_scatter_hidden_states()` 会走：

```python
hidden_states, global_hidden_states = (
    get_local_dp_buffer(),
    hidden_states,
)
...
dp_scatter(hidden_states, global_hidden_states, forward_batch)
```

注意：

```text
dp_scatter 只是本地 memcpy
不是 collective
```

因此在 `SUM_LEN` 路径下，每层后半段的真实 collective 是：

```text
1 次 full TP all-reduce over 8 ranks
```

### 33.2 `MAX_LEN` 路径：跳过 MLP/MoE 的 all-reduce，改走 `dp_reduce_scatter_tensor`

`LayerCommunicator.should_use_reduce_scatter()` 在满足条件时会返回 True：

```python
if (
    self._communicate_summable_tensor_pair_fn
    is CommunicateSummableTensorPairFn._scatter_hidden_states
    and forward_batch.dp_padding_mode.is_max_len()
):
    return True
```

于是模型层会把 `skip_all_reduce` 打开，改由 `postprocess_layer()` 触发：

```python
if allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
    dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
else:
    dp_scatter(hidden_states, global_hidden_states, forward_batch)
```

### 33.3 `dp_reduce_scatter_tensor` 在本节里的真实实现

因为本节：

- `tp_size = 8`
- `attn_dp_size = 4`

所以不会进入最简单的 `tp_size == attn_dp_size` 分支，而会走：

```python
scattered_local_tokens = input.tensor_split(
    get_tensor_model_parallel_world_size()
)[get_tensor_model_parallel_rank()]
get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)
```

也就是说它会变成两步：

#### 第一步

```text
Reduce-Scatter over full TP group of size 8
```

#### 第二步

```text
All-Gather over each ATTN_TP group of size 2
```

### 33.4 这次通信的高层语义

它的目的不是“单纯省一次 all-reduce”，而是：

```text
把 FULL 视图下的 MLP/MoE 输出，重新分发回每个 DP shard 的 TP_ATTN_FULL 形态
```

这样下一层 attention 才能继续只处理自己的那份 token。

### 33.5 本节单层通信数量的两个版本

对每一层来说：

#### `SUM_LEN` 版本

```text
attention 后：1 次 DP gather（底层是 8 卡 TP all-reduce）
MLP/MoE 后：1 次 8 卡 TP all-reduce
```

总计：

```text
每层 2 次 collective
```

#### `MAX_LEN` 版本

```text
attention 后：
  1 次 2 卡 ATTN_TP reduce-scatter
  1 次 8 卡 TP all-gather

MLP/MoE 后：
  1 次 8 卡 TP reduce-scatter
  1 次 2 卡 ATTN_TP all-gather
```

总计：

```text
每层 4 次 collective
```

这正是为什么 DPA decode 往往比普通 TP 路径更依赖通信实现细节。

***

## 34. 场景三的最终 logits：不只是 TP all-gather，还要先做 DP gather，再做本地回散

这一节也和前一版文档差别非常大。

### 34.1 先看初始化：本节默认只开 DPA，不开 DP LM Head

`LogitsProcessor.__init__()` 里：

```python
self.use_attn_tp_group = get_global_server_args().enable_dp_lm_head
...
else:
    self.do_tensor_parallel_all_gather = (
        not skip_all_gather and get_tensor_model_parallel_world_size() > 1
    )
    self.do_tensor_parallel_all_gather_dp_attn = (
        self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1
    )
```

因为本节只说“开启 DPAttention”，没有额外说开启 `enable_dp_lm_head`，所以默认理解为：

```text
enable_dp_attention = True
enable_dp_lm_head = False
```

### 34.2 因此 logits 路径会多出一轮 `dp_gather_replicate`

`_gather_dp_attn_hidden_states()`：

```python
if self.do_tensor_parallel_all_gather_dp_attn:
    logits_metadata.compute_dp_attention_metadata()
    local_hidden_states = hidden_states
    hidden_states = logits_metadata.gathered_buffer
    dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
    return hidden_states, local_hidden_states
```

这一步的语义是：

```text
把各个 DP shard 上最后需要出 logits 的 hidden states 收集起来
```

注意这次是：

```text
dp_gather_replicate
```

而不是前面层间的 `dp_gather_partial`。

它们的差别在于：

- `partial`：只需要合出 full token buffer 供下游继续算
- `replicate`：要让所有相关 rank 都拿到用于 logits 计算的完整视图

### 34.3 然后才是 full TP=8 的 vocab logits all-gather

之后 `_get_logits()` 继续：

```python
logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

if self.do_tensor_parallel_all_gather:
    if self.use_attn_tp_group:
        logits = self._gather_attn_tp_logits(logits)
    else:
        logits = tensor_model_parallel_all_gather(logits)
```

因为 `enable_dp_lm_head=False`，所以这里仍然是：

```text
All-Gather over full TP group of size 8
```

语义与前一版文档一样：

- 每张卡持有 `V/8` 的 vocab shard logits
- 最终要拼回完整 vocab

### 34.4 计算完 logits 后，再回到本地 DP shard

最后 `_scatter_dp_attn_logits()`：

```python
if self.do_tensor_parallel_all_gather_dp_attn:
    global_logits = logits
    logits = torch.empty(
        (local_hidden_states.shape[0], global_logits.shape[1]),
        device=global_logits.device,
        dtype=global_logits.dtype,
    )
    dp_scatter(logits, global_logits, logits_metadata)
```

这一步不是 collective，而是本地切片复制，把全局 logits buffer 中属于当前 DP shard 的部分拿回来。

### 34.5 所以本节 logits 阶段的完整图景是

如果不启用 `dp_lm_head`，则：

```text
DP gather replicate
-> full TP all-gather vocab logits
-> local dp_scatter
```

而不是前一版文档里那种简单的：

```text
直接 TP all-gather
```

### 34.6 如果未来再打开 `enable_dp_lm_head`

那又会进一步变化：

- embedding 已经按 `ATTN_TP`
- LM head 也会改按 `ATTN_TP`
- 最终 logits gather 会改成 `attn_tp_all_gather`

但这已经超出你这次新增要求的范围，所以这里只点到为止。

***

## 35. 场景三的总表：真实发生的集合通信

### 35.1 Embedding 阶段

| 阶段        | 通信组                     | 操作         | 内容                  | 语义                               |
| --------- | ----------------------- | ---------- | ------------------- | -------------------------------- |
| embedding | 4 个 `ATTN_TP` 小组，每组 2 卡 | all-reduce | `[local_tokens, H]` | 合并同一 DP shard 内的 embedding shard |

### 35.2 每层 attention -> MLP 阶段

| padding 模式 | 操作序列                                                        | 语义           |
| ---------- | ----------------------------------------------------------- | ------------ |
| `SUM_LEN`  | 1 次 full TP(8) all-reduce                                   | 完成 DP gather |
| `MAX_LEN`  | 1 次 `ATTN_TP(2)` reduce-scatter + 1 次 full TP(8) all-gather | 完成 DP gather |

### 35.3 每层 MLP/MoE -> 下一层 attention 阶段

| padding 模式 | 操作序列                                                        | 语义                      |
| ---------- | ----------------------------------------------------------- | ----------------------- |
| `SUM_LEN`  | 1 次 full TP(8) all-reduce                                   | 合并 MLP/MoE 输出           |
| `MAX_LEN`  | 1 次 full TP(8) reduce-scatter + 1 次 `ATTN_TP(2)` all-gather | 完成 DP reduce-scatter 回散 |

### 35.4 最终 logits 阶段

| 阶段     | 操作序列                                                               | 语义                                    |
| ------ | ------------------------------------------------------------------ | ------------------------------------- |
| logits | `dp_gather_replicate` + full TP(8) all-gather + local `dp_scatter` | 先收齐 logits 输入，再拼回完整 vocab，再切回本地 shard |

***

## 36. 场景三与前两个场景的并排对比

| 场景                 | 系统结构                                 | 单请求是否跨全部卡     | 每层主通信                                                       | 典型特征           |
| ------------------ | ------------------------------------ | ------------- | ----------------------------------------------------------- | -------------- |
| `TP=8, DP=1`       | 1 个 8 卡 TP world                     | 是             | attention 后 TP all-reduce；MLP/MoE 后 TP all-reduce           | 最标准 TP 路径      |
| `TP=4, DP=2`       | 2 套独立 4 卡 TP 副本                      | 否，只跨某一套 4 卡副本 | attention 后 4 卡 TP all-reduce；MLP/MoE 后 4 卡 TP all-reduce   | 多副本负载均衡，不是 DPA |
| `TP=8, DP=4 + DPA` | 1 个 8 卡 world，`attn_dp=4, attn_tp=2` | 是             | DP gather / DP reduce-scatter + `ATTN_TP` 小组通信 + full TP 通信 | 通信拓扑最复杂        |

***

## 37. 追加总结

把这次追加的两种场景压缩成最核心的两句话：

### 37.1 `TP=4, DP=2`

它在 SGLang 里默认应该理解为：

```text
两套独立的 TP=4 Ascend 推理副本
```

所以：

- 单请求不会同时跨 8 张卡
- 不会出现 DPA collectives
- 通信只是把前一版文档中的 8 卡 TP 通信，缩成 4 卡 TP 通信

### 37.2 `TP=8, DP=4 + enable_dp_attention`

它则是真正的：

```text
一个 8 卡 world 内的 attention 数据并行
```

所以：

- embedding 改成按 `ATTN_TP=2` 小组通信
- attention 与 MLP 之间要做 `DP gather`
- MLP/MoE 与下一层 attention 之间要做 `DP scatter / DP reduce-scatter`
- 最终 logits 也要先做 DPA 相关 gather，再做 full TP vocab gather

如果说前一版文档的通信主旋律是：

```text
TP all-reduce + logits all-gather
```

那么本次追加后，第三个场景的通信主旋律就变成了：

```text
ATTN_TP 小组通信 + FULL TP 通信 + 逻辑 DP gather/scatter 的组合
```

这也是 SGLang Ascend NPU 路径里，**从普通 TP 推理走向更复杂 DPA 推理** 时，最关键的结构性变化。

***

## 38. 三种场景的横向时序总表

这一节把本文分析过的三种场景并排放在一起。目的不是重复前文，而是回答一个非常实用的问题：

```text
同一个请求，在三种部署形态里，通信时序到底哪里一样，哪里不一样
```

三种场景分别是：

1. `TP=8, DP=1, EP=1`
2. `TP=4, DP=2`，默认不开 `enable_dp_attention`
3. `TP=8, DP=4, enable_dp_attention=True`

### 38.1 系统组织总览表

| 维度             | `TP=8, DP=1`  | `TP=4, DP=2`     | `TP=8, DP=4 + DPA`                    |
| -------------- | ------------- | ---------------- | ------------------------------------- |
| 物理卡使用方式        | 1 个 8 卡 world | 2 套独立 4 卡副本      | 1 个 8 卡 world                         |
| 单请求跨卡范围        | 跨 8 卡         | 只跨其中 1 套 4 卡副本   | 跨 8 卡                                 |
| 模型内部 `dp_size` | 1             | 1                | 4                                     |
| `attn_tp_size` | 8             | 4                | 2                                     |
| `attn_dp_size` | 1             | 1                | 4                                     |
| embedding 通信组  | TP(8)         | TP(4)            | `ATTN_TP(2)`                          |
| 每层 attention 后 | TP all-reduce | TP all-reduce    | DP gather                             |
| 每层 MLP/MoE 后   | TP all-reduce | TP all-reduce    | DP scatter / DP reduce-scatter        |
| logits 汇总      | TP all-gather | TP all-gather    | `dp_gather_replicate` + TP all-gather |
| 通信复杂度风格        | 标准 TP         | 标准 TP，但 group 更小 | DPA 组合通信                              |

### 38.2 从请求进入到输出返回的横向时序表

| 阶段                            | `TP=8, DP=1`      | `TP=4, DP=2`      | `TP=8, DP=4 + DPA`                        |
| ----------------------------- | ----------------- | ----------------- | ----------------------------------------- |
| 请求进入 controller               | 进入唯一副本            | 被路由到某个 4 卡副本      | 进入唯一 8 卡 DPA world                        |
| Scheduler 组批                  | 单套 scheduler      | 某个副本内 scheduler   | 单套 DPA-aware scheduler                    |
| Embedding                     | 8 卡 TP all-reduce | 4 卡 TP all-reduce | 4 个 2 卡 `ATTN_TP` all-reduce              |
| Layer 内 attention 本体          | 本地 NPU attention  | 本地 NPU attention  | 本地 NPU attention                          |
| Attention 后进入 MLP             | 8 卡 TP all-reduce | 4 卡 TP all-reduce | DP gather                                 |
| MLP / MoE 计算                  | full hidden 上本地算  | full hidden 上本地算  | FULL 视图上本地算                               |
| MLP / MoE 后回到下一层 attention 输入 | 8 卡 TP all-reduce | 4 卡 TP all-reduce | DP scatter 或 DP reduce-scatter            |
| Final logits                  | 8 卡 TP all-gather | 4 卡 TP all-gather | `dp_gather_replicate` + 8 卡 TP all-gather |
| 输出采样                          | 本地采样              | 本地采样              | 本地采样                                      |

### 38.3 单层 decoder 的横向对比表

| 单层阶段           | `TP=8, DP=1`          | `TP=4, DP=2`          | `TP=8, DP=4 + DPA`                               |
| -------------- | --------------------- | --------------------- | ------------------------------------------------ |
| 输入形态           | FULL                  | FULL                  | `TP_ATTN_FULL` / shard 化                         |
| `self_attn` 之后 | partial hidden        | partial hidden        | attention shard / `TP_ATTN_FULL`                 |
| 进入 MLP 前       | TP all-reduce -> FULL | TP all-reduce -> FULL | DP gather -> FULL                                |
| MLP / MoE 输出   | partial hidden        | partial hidden        | FULL 或待回散的输出                                     |
| 层结束时           | TP all-reduce -> FULL | TP all-reduce -> FULL | DP scatter / DP reduce-scatter -> `TP_ATTN_FULL` |

### 38.4 从“通信原语个数”角度的横向总结

假设忽略极端优化分支，只看主线：

| 场景                 | embedding                | 每层主通信                                                       | logits                                | 主线直觉      |
| ------------------ | ------------------------ | ----------------------------------------------------------- | ------------------------------------- | --------- |
| `TP=8, DP=1`       | 1 次 TP all-reduce        | 2 次 TP all-reduce                                           | 1 次 TP all-gather                     | 最规整       |
| `TP=4, DP=2`       | 1 次 4 卡 TP all-reduce    | 2 次 4 卡 TP all-reduce                                       | 1 次 4 卡 TP all-gather                 | 与标准 TP 同构 |
| `TP=8, DP=4 + DPA` | 1 次 `ATTN_TP` all-reduce | `SUM_LEN` 时每层约 2 次 collective；`MAX_LEN` 时每层约 4 次 collective | 至少 1 次 DPA gather + 1 次 TP all-gather | 通信最复杂     |

### 38.5 一句话抓住三者区别

你可以把三种场景压缩成下面三句话：

#### 场景 A：`TP=8, DP=1`

```text
标准 8 卡 TP 推理，通信主旋律是 TP all-reduce + logits all-gather
```

#### 场景 B：`TP=4, DP=2`

```text
不是单 world 的 DP，而是两套独立 TP=4 副本；单请求只在某一套 4 卡里通信
```

#### 场景 C：`TP=8, DP=4 + DPA`

```text
真正的 attention 数据并行；通信从“纯 TP 汇总”升级为“ATTN_TP + FULL TP + 逻辑 DP gather/scatter”的组合
```

***

## 39. 按 Prefill / Decode 分开的逐步通信流水图

这一节把三种场景都按：

- Prefill
- Decode

分别展开。这样可以帮助你回答两个线上最常见的问题：

1. **一次请求第一次进来时，到底怎么通信**
2. **生成后续每个 token 时，到底怎么重复通信**

### 39.1 场景一：`TP=8, DP=1, EP=1`

#### 39.1.1 Prefill 逐步通信流水图

```text
请求进入
-> tokenizer / scheduler / batch 组织
-> 进入 8 卡 TP world
-> Embedding 本地查表
-> TP(8) all-reduce
   含义：把 vocab shard embedding 合并成完整 hidden
-> Layer 0 attention 本地 NPU 计算
-> TP(8) all-reduce
   含义：合并 attention o_proj 的 partial hidden
-> Layer 0 MLP 或 MoE 本地计算
-> TP(8) all-reduce
   含义：合并 MLP/MoE 输出
-> Layer 1
-> TP(8) all-reduce
-> TP(8) all-reduce
-> ...
-> Layer L-1
-> TP(8) all-reduce
-> TP(8) all-reduce
-> LM Head 计算局部 vocab logits
-> TP(8) all-gather
   含义：把 [T_prefill, V/8] 拼回 [T_prefill, V]
-> sampling / 输出首 token 或相关结果
```

#### 39.1.2 Decode 逐步通信流水图

```text
请求进入某一轮 decode
-> Scheduler 组织 decode batch
-> 进入 8 卡 TP world
-> Embedding 本地查表
-> TP(8) all-reduce
   含义：恢复本轮 token 的完整 hidden
-> Layer 0 attention 本地 NPU 计算
   注：读取 KV cache，但这不是 collective
-> TP(8) all-reduce
-> Layer 0 MLP 或 MoE 本地计算
-> TP(8) all-reduce
-> ...
-> Layer L-1
-> TP(8) all-reduce
-> TP(8) all-reduce
-> LM Head 局部 logits
-> TP(8) all-gather
-> sampling
-> 生成 1 个新 token
```

#### 39.1.3 场景一最值得记住的点

- prefill 和 decode 的通信种类完全相同
- 主要区别只在于：
  - prefill 的张量更大
  - decode 的通信轮数更多

***

### 39.2 场景二：`TP=4, DP=2`，默认不开 DPA

#### 39.2.1 Prefill 逐步通信流水图

```text
请求进入
-> controller 负载均衡
-> 被路由到某一个 4 卡副本
-> tokenizer / scheduler / batch 组织
-> 进入该副本内部的 TP(4) world
-> Embedding 本地查表
-> TP(4) all-reduce
   含义：合并该副本内 1/4 vocab shard 的 embedding 输出
-> Layer 0 attention 本地 NPU 计算
-> TP(4) all-reduce
-> Layer 0 MLP 或 MoE 本地计算
-> TP(4) all-reduce
-> ...
-> Layer L-1
-> TP(4) all-reduce
-> TP(4) all-reduce
-> LM Head 局部 logits
-> TP(4) all-gather
-> sampling / 输出结果
```

#### 39.2.2 Decode 逐步通信流水图

```text
请求进入某一轮 decode
-> 请求仍然停留在同一个 4 卡副本里
-> Scheduler 组织 decode batch
-> Embedding 本地查表
-> TP(4) all-reduce
-> Layer 0 attention 本地 NPU 计算
-> TP(4) all-reduce
-> Layer 0 MLP 或 MoE 本地计算
-> TP(4) all-reduce
-> ...
-> Layer L-1
-> TP(4) all-reduce
-> TP(4) all-reduce
-> TP(4) all-gather
-> sampling
-> 输出当前步 token
```

#### 39.2.3 场景二最值得记住的点

- 从单请求视角看，它几乎就是“缩小版的场景一”
- 真正的区别不是通信原语变了，而是：
  - group size 从 8 变成 4
  - 系统里多出了一套不参与当前请求的独立副本

***

### 39.3 场景三：`TP=8, DP=4, enable_dp_attention=True`

这一节是最复杂、也是最值得仔细读的一节。

#### 39.3.1 Prefill 逐步通信流水图

在 prefill 场景中，由于不同 DP shard 的 token 数往往不一致，更常见的是 `SUM_LEN` 风格的通信语义。它的高层图景可以写成：

```text
请求进入
-> tokenizer / scheduler / batch 组织
-> 进入唯一的 8 卡 DPA world
-> 按 attn_dp=4, attn_tp=2 组织输入
-> Embedding 本地查表
-> 每个 ATTN_TP(2) 小组做 all-reduce
   含义：合并该 DP shard 内的 embedding shard
-> Layer 0 attention 本地 NPU 计算
-> dp_gather_partial
   常见实现语义：full TP(8) all-reduce
   含义：把 4 个 DP shard 的 token 聚成 FULL 视图，供 MLP/MoE 使用
-> Layer 0 MLP 或 MoE 本地计算
-> full TP(8) all-reduce
   含义：合并 FULL 视图上的 MLP/MoE 输出
-> dp_scatter
   含义：把 FULL 输出切回当前 DP shard 的 attention 输入形态
-> Layer 1 attention
-> dp_gather_partial
-> full TP(8) all-reduce
-> dp_scatter
-> ...
-> Layer L-1
-> dp_gather_partial
-> full TP(8) all-reduce
-> dp_scatter
-> logits 前的 dp_gather_replicate
   含义：把需要出 logits 的 hidden 收齐
-> LM Head 局部 vocab logits
-> full TP(8) all-gather
   含义：拼回完整 vocab logits
-> local dp_scatter
-> sampling / 输出结果
```

如果你想抓主线，可以把 prefill 的单层结构压缩成：

```text
TP_ATTN_FULL
-> DP gather
-> FULL
-> MLP/MoE
-> FULL TP 汇总
-> 回到 DP shard
```

#### 39.3.2 Decode 逐步通信流水图

decode 时，DPA 更容易进入 `MAX_LEN` 风格的规整路径。此时单层通信图景会更像“拆成两段小组 + 全组”的组合：

```text
请求进入某一轮 decode
-> Scheduler 组织 decode batch
-> 进入 8 卡 DPA world
-> Embedding 本地查表
-> 每个 ATTN_TP(2) 小组 all-reduce
-> Layer 0 attention 本地 NPU 计算
-> dp_gather_partial
   典型 MAX_LEN 实现：
   1) ATTN_TP(2) reduce-scatter
   2) full TP(8) all-gather
   含义：把 attention shard 聚成 FULL
-> Layer 0 MLP 或 MoE 本地计算
-> dp_reduce_scatter_tensor
   典型实现：
   1) full TP(8) reduce-scatter
   2) ATTN_TP(2) all-gather
   含义：把 FULL 输出重新回散成下一层 attention 所需的 shard 形态
-> Layer 1
-> 再重复：
   attention
   -> DP gather 组合通信
   -> MLP/MoE
   -> DP reduce-scatter 组合通信
-> ...
-> Layer L-1
-> logits 前 dp_gather_replicate
-> full TP(8) all-gather vocab logits
-> local dp_scatter
-> sampling
-> 生成本轮 token
```

#### 39.3.3 场景三最值得记住的点

- DPA 不是简单“把 TP all-reduce 换个名字”
- 它真正改变了：
  - 输入输出张量布局
  - attention 与 MLP 之间的数据形态
  - 通信所在的小组大小
  - 单层 collective 的数量与组合方式

***

## 40. 用最简洁的话复盘三种场景的 Prefill / Decode 区别

如果你读完整篇文档后，只想保留最核心的心智图，可以记下面这 6 句话：

### 40.1 对三种场景都成立的一条总规律

```text
Prefill 和 Decode 的“通信种类”通常比你想象得更接近
真正差别更多体现在 token 数量、张量大小、步数和是否进入规整 padding 路径
```

### 40.2 `TP=8, DP=1`

```text
Prefill = 大张量 TP all-reduce / all-gather
Decode  = 高频小张量 TP all-reduce / all-gather
```

### 40.3 `TP=4, DP=2`

```text
对单请求而言，本质上是“缩小通信组后的标准 TP 推理”
```

### 40.4 `TP=8, DP=4 + DPA`

```text
Prefill 往往更像“DP gather + FULL TP 汇总 + 回散”
Decode 往往更像“ATTN_TP 小组通信 + FULL TP 通信”的组合流水
```

### 40.5 为什么线上最难调的是第三种场景

因为它同时引入了：

- 逻辑 DP 维度
- `ATTN_TP` 小组
- FULL TP 组
- `SUM_LEN / MAX_LEN` 两类不同通信路径

所以问题排查时，不能只看：

```text
是不是 all-reduce 慢
```

而要看：

- 慢的是哪一段 collective
- 是 `ATTN_TP(2)` 的小组通信，还是 `TP(8)` 的全组通信
- 是 prefill 的 `SUM_LEN` 路径，还是 decode 的 `MAX_LEN` 路径

### 40.6 最后的最简版结论

如果把本文所有场景再压缩到一句话：

```text
标准 TP 场景的核心是“partial -> full”的 all-reduce / all-gather；
DPA 场景的核心则是“shard <-> full”之间反复切换所带来的组合通信。
```

***

## 41. 公开资料补充说明

本次新增的“集合通信基础概念”和“HCCL 底层原理”部分，除了结合 SGLang 源码理解之外，还参考了公开资料中的如下关键信息：

- HCCL 是昇腾平台上的高性能集合通信库，支持 AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll 等原语
- HCCL 的公开架构分为通信框架、通信算法、通信平台三层
- HCCL 支持 Mesh、Ring、Halving-Doubling / RHD、PairWise、Star、Pipeline 等算法
- HCCL 可运行在 HCCS、RoCE、PCIe 等链路之上
- HCCL 公共资料中给出了 α–β–γ 性能模型与 Runtime 下发任务到设备执行的说明
- Ascend C HCCL 高阶 API 公开资料展示了 Prepare / Commit / Wait / Finalize 这类更底层的通信任务编排方式

