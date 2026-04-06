# SGlang 仓库 NPU 后端 KVCache 详细解析与流程梳理

本指南以 SGlang 仓库中的 NPU (华为 Ascend 昇腾) 后端为例，逐行、细致地梳理在大语言模型 (LLM) 推理请求生成和执行过程中，与 KVCache 相关的所有核心逻辑。无论您是刚接触大模型推理的新手，还是想要深入了解底层实现优化的开发者，都可以通过本文建立起完整的认知。

---

## 1. 总体流程概述 (Overview)

在大模型推理中，生成文本分为两个主要阶段：
1. **Prefill (预填充/Extend 阶段)**：一次性处理用户输入的所有 Prompt tokens，计算出对应的 Key (K) 和 Value (V) 向量，并将它们存入显存 (KVCache) 中。
2. **Decode (解码阶段)**：模型逐个生成新的 token。每次生成新 token 时，只需计算当前 token 的 K 和 V 并存入 KVCache，然后结合之前存好的 KVCache 进行 Attention 计算，避免了重复计算，极大提升了推理速度。

在 SGlang 中，KVCache 的管理采用 **PagedAttention (分页注意力)** 的思想：
- 内存不再为每个请求分配连续的整块空间，而是切分为大小固定的 **Page (页/块)**。
- 采用 **Token -> Slot (槽位)** 的映射机制，请求的 Token 会被动态分配到空闲的 Page 中。
- NPU 后端对 KVCache 进行了深度定制，使用了华为 `torch_npu` 库中的高性能算子（如 FIA, Paged Attention 等）进行读写和计算。

**核心模块：**
- **Data Structure (数据结构)**：`NPUMHATokenToKVPool` / `NPUMLATokenToKVPool` 负责底层的显存布局和读写。在系统启动时初始化。
- **Allocator (分配器)**：`NPUPagedTokenToKVPoolAllocator` 负责逻辑上的 Page/Slot 槽位分配。在系统调度请求阶段被调用。
- **Attention Backend (计算后端)**：`AscendAttnBackend` 负责在模型前向传播中，把分配好的 KVCache 喂给 NPU 底层算子。在模型执行阶段被调用。

---

## 2. KVCache 的底层数据结构 (Memory Pool)

显存的物理管理主要在 `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` 文件中实现。这里区分了标准的 MHA (Multi-Head Attention) 和 MLA (Multi-head Latent Attention，如 DeepSeek V2/V3 使用)。

### 2.1 MHA 的存储结构：`NPUMHATokenToKVPool`
**在请求处理整体流程中的位置：**
系统冷启动初始化阶段。当 SGLang 服务器启动，并检测到配置为 NPU 后端且未使用 MLA 模型时，在 `ModelRunner.init_kv_cache` 方法中被实例化。它提前占用 NPU 的显存，之后在整个服务器生命周期内保持常驻。

当使用普通的注意力机制时，SGlang 会在初始化时预先在 NPU 上开辟一整块显存用于 KVCache。

```python
# [memory_pool_npu.py] NPUMHATokenToKVPool._create_buffers 
self.kv_buffer = torch.zeros(
    (
        2,                               # 0 存 Key, 1 存 Value
        self.layer_num,                  # 模型的层数 (Layers)
        self.size // self.page_size + 1, # 总的 Page 数量 (+1 是为了 padding)
        self.page_size,                  # 每个 Page 包含的 Token 数量 (通常为 16 或 64)
        self.head_num,                   # 注意力头数
        self.head_dim,                   # 每个头的维度大小
    ),
    dtype=self.store_dtype,
    device=self.device,
)
```
**细节解释：**
- 这里的内存布局是**连续的 (Continuous)**。连续内存的设计能够极大提高 Ascend 昇腾底层 DMA (直接内存访问) 和计算后端的传输效率。
- 第一个维度 `2` 将 Key 和 Value 打包在一起。随后可以通过 `self.k_buffer = self.kv_buffer[0]` 和 `self.v_buffer = self.kv_buffer[1]` 分别获取。

**写入 KVCache (Prefill/Decode 产生新 token 时)：**
当模型计算出新的 K/V 后，需要写入缓存。SGlang 提供了 `set_kv_buffer` 方法：
```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
    # loc 是分配器分配的物理槽位索引
    if self.use_fia:
        # 如果开启了 FIA (Fused Infer Attention) 高性能模式
        # 使用 npu_scatter_nd_update_ 算子进行高效稀疏更新
        torch_npu.npu_scatter_nd_update_(
            k_buffer_layer, loc.view(-1, 1), cache_k.view(-1, 1, self.head_num, self.head_dim)
        )
    else:
        # 普通模式下，使用昇腾专用的 reshape_and_cache 算子
        torch_npu._npu_reshape_and_cache(
            key=cache_k,
            value=cache_v,
            key_cache=self.k_buffer[...],
            value_cache=self.v_buffer[...],
            slot_indices=loc.to(torch.int32),
        )
```

### 2.2 MLA 的存储结构：`NPUMLATokenToKVPool`
**在请求处理整体流程中的位置：**
系统冷启动初始化阶段。与 MHA 类似，但当检测到模型使用了 DeepSeek 类型的 MLA 架构时，在 `ModelRunner.init_kv_cache` 方法中被实例化。

针对 DeepSeek 等使用 MLA 的模型，K 和 V 不再是完整的矩阵，而是被压缩后的 latent vector 和 RoPE 特征。因此它将 buffer 拆分：
- `self.k_buffer`: 形状为 `(layer_num, num_pages, page_size, 1, kv_lora_rank)`
- `self.v_buffer`: 存储解耦后的 ROPE 信息，形状为 `(layer_num, num_pages, page_size, 1, qk_rope_head_dim)`
- 写入时统一使用 `torch_npu.npu_scatter_nd_update_` 进行稀疏张量更新。

---

## 3. 槽位分配机制 (Slot Allocation)

虽然显存池建立好了，但当一个请求过来时，系统怎么知道该把 Token 写到 `kv_buffer` 的哪个位置呢？这由 `NPUPagedTokenToKVPoolAllocator` 负责 (`allocator_npu.py`)。它维护着一个空闲页列表 `free_pages`。

### 3.1 刚开始的槽位分配：Prefill 阶段 (`alloc_extend`)
**在请求处理整体流程中的位置：**
当新的请求进入系统，调度器 (Scheduler) 决定执行预填充 (Prefill) 时触发。
调用链：`Scheduler.get_new_batch_prefill` -> `ScheduleBatch.prepare_for_extend` -> 最终调用 Allocator 的 `alloc_extend`。此时，系统尚未开始底层 NPU 上的模型前向传播，而是先在调度层面为这些即将进入模型的 Token 预留好物理显存位置。

当新请求到达，包含多个 Token 时，需要分配多个连续或不连续的槽位。

```python
# [allocator_npu.py] NPUPagedTokenToKVPoolAllocator.alloc_extend
def alloc_extend(self, prefix_lens, seq_lens, last_loc, extend_num_tokens):
    # 1. 计算需要的 Page 数量
    num_new_pages = (
        (seq_lens + self.roundup) // self.page_size
        - (prefix_lens + self.roundup) // self.page_size
    ).sum()
    
    # ... 如果剩余页不够，触发整理或直接返回 None 等待 ...

    # 2. 分配槽位 (Slot Indices)
    if num_new_pages_item < 200:
        # 【NPU 专有优化】: 如果请求较小，直接调用自定义的 NPU kernel 在显存中并行分配
        from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
        alloc_extend_kernel[...] # 计算出 out_indices
    else:
        # 回退到通用的 CPU/Naive 分配算法
        alloc_extend_naive[...]

    # 3. 更新空闲页列表
    self.free_pages = self.free_pages[num_new_pages_item:]
    return out_indices.int()
```
**细节解释：**
- `out_indices` 返回的是一个 1D Tensor，里面记录了这批新 tokens 应该存放在物理 KVCache 中的绝对槽位号 (例如 `[1024, 1025, 1026, 3012, 3013...]`)。这些索引后续会作为 `loc` 传给前面的 `set_kv_buffer`。

### 3.2 过程中的槽位分配：Decode 阶段 (`alloc_decode`)
**在请求处理整体流程中的位置：**
在模型完成 Prefill 阶段并吐出第一个 Token 后，进入逐字生成的 Decode 循环。每次生成新 Token 前触发。
调用链：`Scheduler.get_next_batch` -> `ScheduleBatch.prepare_for_decode` -> 最终调用 Allocator 的 `alloc_decode`。系统每步循环仅需为新生成的 1 个 Token 分配槽位。

Decode 阶段每次只生成 1 个新 Token。

```python
def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
    # 计算是否跨越了 Page 边界
    num_new_pages = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True)
    
    # 检查当前 token 是不是正好是新 page 的第一个 token
    need_new_pages = (seq_lens % self.page_size == 1).int()
    
    # 如果不需要新 page，就在 last_loc (上一个 token 的位置) 往后加 1
    # 如果需要新 page，就从 free_pages 中取出一个新 page 的起始位置
    out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[...] * self.page_size * need_new_pages
    
    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices.int()
```

---

## 4. 请求处理整体流程与 Attention 计算

在实际执行中，从请求调度到硬件执行的完整流转如下，主要在 `ascend_backend.py` (`AscendAttnBackend` 类) 中实现：

### Step 1: Scheduler 准备 `ForwardBatch`
**在请求处理整体流程中的位置：**
当调度器确定了本轮要处理的请求集合后（不论是 Prefill 还是 Decode），会把这些请求的序列信息、刚才分配器给出的槽位信息 (`out_cache_loc`) 等打包成 `ForwardBatch`，准备传递给底层的模型执行器 (`ModelRunner`)。

### Step 2: KVCache 元数据初始化 (`init_forward_metadata`)
**在请求处理整体流程中的位置：**
进入模型的前向传播 (`ModelRunner.forward`) 后，在执行具体的 Transformer 层之前，注意力后端 (`AscendAttnBackend`) 会被调用以初始化这一轮批处理的页表。

在每层 Attention 计算前，先准备 Block Table (页表)。
```python
# 将全局的 Token 映射池转换为当前 batch 的 Block Tables
self.forward_metadata.block_tables = (
    forward_batch.req_to_token_pool.req_to_token[
        forward_batch.req_pool_indices, :seq_lens_max
    ][:, :: self.page_size] // self.page_size
)
```
这就类似操作系统的虚拟内存页表，记录了请求逻辑上的 Sequence 是如何映射到物理分散的 Pages 上的。

### Step 3: Prefill 阶段 Attention 计算 (`forward_extend`)
**在请求处理整体流程中的位置：**
模型前向传播期间。当模型在预填充阶段计算完 Q、K、V 的投影后，在注意力层 (`RadixAttention.forward`) 中调用后端的 `forward_extend`。此时，新的 K/V 被正式写入之前调度器分配的物理显存槽位，并执行首次的全量注意力计算。

在处理长文本 Prompt 时：
1. **保存 KVCache**: 
   ```python
   # 把计算出的 k, v 写入显存池，写入位置为 allocator 分配的 out_cache_loc
   if save_kv_cache and k is not None and v is not None:
       forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
   ```
2. **获取全局 Cache 指针**:
   ```python
   k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
   v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
   ```
3. **调用 NPU 算子计算 Attention**:
   SGlang 在 Ascend 上优先使用 **FIA (Fused Infer Attention)**：
   ```python
   if self.use_fia:
       # 循环处理每个请求的 query，使用 npu_fused_infer_attention_score 进行全量注意力计算
       attn_output[...] = torch.ops.npu.npu_fused_infer_attention_score(
           q[...], k[...], v[...], 
           num_heads=layer.tp_q_head_num, 
           atten_mask=self.fia_mask.unsqueeze(0), # NPU 特有的下三角 Mask 矩阵
           ...
       )
   else:
       # 降级使用 _npu_flash_attention_qlens
       torch_npu._npu_flash_attention_qlens(
           query=query, key_cache=k_cache, value_cache=v_cache,
           block_table=self.forward_metadata.block_tables, ...
       )
   ```

### Step 4: Decode 阶段 Attention 计算 (`forward_decode`)
**在请求处理整体流程中的位置：**
模型前向传播期间。与 Prefill 类似，但发生在增量解码循环的注意力层中 (`RadixAttention.forward`)。每次生成一个 Token 时，都会调用后端的 `forward_decode` 将当前 Token 的 K/V 写入刚分配的新槽位（或紧接着上一个 Token 的槽位），并执行增量的 Paged Attention 计算。

在逐字生成时：
1. **保存单个 KVCache**: 同样调用 `set_kv_buffer` 将新生成的 1 个 token 的 K/V 存入缓存。
2. **调用 Paged Attention 算子**:
   这里不再需要把所有的 K/V 传给算子，而是只传 `query`、`k_cache` 完整显存池、以及 `block_tables` (页表)。NPU 底层算子会根据页表自动去显存里捞出对应的历史 K/V 进行计算。
   ```python
   # 使用 NPU 原生的 Paged Attention 算子
   torch_npu._npu_paged_attention(
       query=query,
       key_cache=k_cache,
       value_cache=v_cache,
       num_heads=layer.tp_q_head_num,
       num_kv_heads=layer.tp_k_head_num,
       block_table=self.forward_metadata.block_tables, # 核心：传入页表
       context_lens=self.forward_metadata.seq_lens_cpu_int,
       out=attn_output,
   )
   ```
   **注：** 如果是 DeepSeek 等 MLA 模型，则调用专为 MLA 优化的 `torch_npu._npu_paged_attention_mla`。

---

## 5. 涉及到的核心底层算子 (Operators) 总结

在整个 KVCache 交互流中，SGlang 深度依赖华为昇腾 `torch_npu` 的底层算子：

1. **缓存写入与更新**：
   - `torch_npu._npu_reshape_and_cache`: MHA 标准模式下的 Cache 写入算子。
   - `torch_npu.npu_scatter_nd_update_`: MLA 或 FIA 模式下的稀疏张量更新算子，支持非连续内存的高效写入。
2. **内存分配**：
   - `sgl_kernel_npu.mem_cache.allocator.alloc_extend_kernel`: SGlang 团队用 NPU Kernel 写的显存端极速 Slot 分配器。
3. **注意力计算 (包含读取 KVCache)**：
   - `torch_npu.npu_fused_infer_attention_score` (FIA): NPU 上极其强大的融合推理算子，同时支持 Prefill 和 Decode，性能极高。
   - `torch_npu._npu_flash_attention_qlens`: 变长序列的 FlashAttention。
   - `torch_npu._npu_paged_attention`: NPU 原生的分页注意力算子，用于 Decode 阶段。
   - `torch_npu._npu_paged_attention_mla`: NPU 原生的 MLA 架构分页注意力算子。

---

## 总结

SGlang 在 NPU 上的 KVCache 流程设计非常严谨：
从最底层的**连续张量池 (`Memory Pool`)** 出发，到上层基于**分页映射的调度分配器 (`Allocator`)** 为每个 Token 指派唯一的物理槽位；在请求执行期间，前向传播层 (`Model Runner`) 获取新 Token 的 K/V，将其写入分配的槽位，最后结合 `block_tables` (页表) 投递给 **NPU 深度定制的 Attention 算子 (`AscendAttnBackend`)**。这种软硬件结合的设计，最大化了 Ascend 昇腾硬件在变长序列和并发请求下的显存带宽与计算利用率。