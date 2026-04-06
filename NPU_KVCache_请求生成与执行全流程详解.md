# SGLang（NPU路径）请求生成与执行中的 KVCache 全流程详解

## 0. 文档范围与阅读方式

- 目标：以 `sglang-main` 仓库的 **NPU/Ascend 路径**为主，梳理“请求进入 -> 调度 -> KV 槽位分配 -> 前向读写 KV -> 生成输出”的完整链路。
- 粒度：对关键函数给出“接近逐行”的解释（按代码段逐行语义展开），重点解释 KVCache 相关数据结构与算子调用。
- 说明：本文主要基于仓库代码行为做静态分析；个别 NPU 私有算子的底层实现细节不在本仓库中，按调用语义解释。

---

## 1. 先看全局：一条生成请求如何走到 KVCache

## 1.1 HTTP 入口到 tokenizer

1. `/generate` 请求进入 FastAPI 路由：  
   `python/sglang/srt/entrypoints/http_server.py:669-709`
2. 路由把请求交给 `TokenizerManager.generate_request`：  
   `python/sglang/srt/managers/tokenizer_manager.py:481-529`
3. 该函数完成规范化、校验、分词后，调用 `_send_one_request` 将 tokenized 请求发给 scheduler：  
   `python/sglang/srt/managers/tokenizer_manager.py:1097-1105`

> 这一步还没有分配 KVCache，只是把“文本请求”变成“token + 采样参数 + 元信息”的内部请求对象。

## 1.2 scheduler 主循环与批次执行

1. scheduler 主循环 `event_loop_normal` 持续执行：收请求 -> 处理请求 -> 取下一批 -> 执行批次 -> 处理结果  
   `python/sglang/srt/managers/scheduler.py:1244-1270`
2. 收到请求后 `handle_generate_request` 把请求包装为 `Req`，入等待队列：  
   `python/sglang/srt/managers/scheduler.py:1611-1845`
3. 真正执行时走 `run_batch`，内部调用 `model_worker.forward_batch_generation`：  
   `python/sglang/srt/managers/scheduler.py:2438-2587`

## 1.3 forward 执行到 attention

1. `TpModelWorker.forward_batch_generation` 会把 `ModelWorkerBatch` 转成 `ForwardBatch`：  
   `python/sglang/srt/managers/tp_worker.py:447-538`
2. `ModelRunner.forward` 根据 forward_mode 选择 `forward_extend` 或 `forward_decode`：  
   `python/sglang/srt/model_executor/model_runner.py:2461-2593`
3. 模型层中 `RadixAttention.forward` 最终调用 `forward_batch.attn_backend.forward(...)`：  
   `python/sglang/srt/layers/radix_attention.py:99-135`
4. NPU 场景这里就是 `AscendAttnBackend`，并在其中完成 KV 写入与读取：  
   `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

---

## 2. NPU 路径为什么会走到 Ascend 后端

## 2.1 参数默认值切到 ascend

`ServerArgs._handle_npu_backends`（`server_args.py:1018-1030`）在 `device == "npu"` 时调用：

- `set_default_server_args(self)`（`hardware_backend/npu/utils.py:39-85`）
  - 强制 `attention_backend/prefill/decode_attention_backend = "ascend"`
  - 默认 `page_size = 128`（如果用户没设置）
  - 设置 NPU 上的 chunked prefill / graph batch size 默认值

## 2.2 NPU 后端初始化

`model_runner.py:202-205` 在 `_is_npu` 时调用 `init_npu_backend()`（`npu/utils.py:86-104`）：

- 导入 `torch_npu` 与 `sgl_kernel_npu`
- 设置 NPU 编译模式
- 做设备相关修正

## 2.3 attention backend 注册

`attention_registry.py:71-77`：

- `"ascend"` -> `AscendAttnBackend`

所以后续 `RadixAttention` 的 `attn_backend.forward` 就会进入 NPU 的 attention 实现。

---

## 3. KVCache 的核心数据结构（NPU）

## 3.1 两类池：MHA 与 MLA

在 `model_runner_kv_cache_mixin.py:441-483`：

- 非 MLA：`NPUMHATokenToKVPool`
- MLA：`NPUMLATokenToKVPool`

两者都位于：`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`

---

## 3.2 `NPUMHATokenToKVPool` 逐段解释

### A) 初始化与 `_create_buffers`（`memory_pool_npu.py:51-84`）

- L57-L68：分配 `kv_buffer`，形状是  
  `[2, layer_num, page_num(+1), page_size, head_num, head_dim]`
  - 第 0 维 `2`：分别放 K 和 V
  - `+1` 是给“填充槽位 0”预留（注释里写明用于 padded token）
- L69-L70：`self.k_buffer = self.kv_buffer[0]`，`self.v_buffer = self.kv_buffer[1]`
- L72-L84：如果启用 `ASCEND_USE_FIA`，把每层 reshape 成 FIA 更友好的视图

### B) `set_kv_buffer`（`memory_pool_npu.py:112-165`）

- L122-L125：确定 layer_id（可被 `layer_id_override` 覆盖）
- L126-L133：若输入 `cache_k/cache_v` dtype 与池 dtype 不同，先做缩放/类型转换
- L134-L137：若存储 dtype 与计算 dtype 不同，再 view 到存储 dtype
- 分支 1（L138-L151，FIA）：
  - 使用 `torch_npu.npu_scatter_nd_update_`
  - 按 `loc`（槽位索引）把当前 token 的 K/V 写到缓存
- 分支 2（L153-L164，非 FIA）：
  - 调用 `torch_npu._npu_reshape_and_cache`
  - 语义是“把当前步 K/V 按 slot_indices 重排并写入 paged cache”

---

## 3.3 `NPUMLATokenToKVPool` 逐段解释

### A) 构造（`memory_pool_npu.py:200-237`）

- `k_buffer` 形状：`[layer_num, page_num(+1), page_size, 1, kv_lora_rank]`
- `v_buffer` 形状：`[layer_num, page_num(+1), page_size, 1, qk_rope_head_dim]`
- 可选 `index_k_buffer`（NSA 场景）

### B) 读取接口

- `get_kv_buffer`（L254-L260）：返回 `(k_buffer[layer], v_buffer[layer])`
- `get_key_buffer/get_value_buffer`（L262-L277）：按需要转 dtype 后返回
- `get_index_k_buffer`（L278-L285）：返回 index_k

### C) 写入接口

- `set_kv_buffer`（L310-L342）：
  - 统一用 `npu_scatter_nd_update_` 按 `loc` 写 `k_buffer/v_buffer`
  - 若 `cache_v is None`，会从 `cache_k` 里按最后一维切分
- `set_index_k_buffer`（L344-L362）：
  - 同样按 `loc` scatter 更新 index_k

---

## 3.4 请求到 token 的索引表：`ReqToTokenPool`

`python/sglang/srt/mem_cache/memory_pool.py:126-188`（定义）  
作用：保存“某个请求第 i 个 token -> KV 全局槽位号”的映射表 `req_to_token`。

这个映射是后续 decode 时寻找历史 KV 的关键。

---

## 4. “刚开始槽位分配”是怎么做的（重点）

这里分两类：**extend/prefill** 与 **decode**。

---

## 4.1 Extend 阶段：`alloc_for_extend`（`mem_cache/common.py:328-392`）

按代码顺序解释：

1. L340：`maybe_evict_swa()`，先做窗口外缓存回收（若启用 SWA）
2. L342：收集每个请求已有的 prefix 索引 `prefix_tensors`
3. L345-L349：构造 prefix/extend 长度的 CPU 与 device tensor
4. L351-L356：调用 `alloc_req_slots` 为请求本身分配 `req_pool_idx`
5. L358-L374：分配 token 对应的 KV 槽位
   - `page_size == 1` 走普通线性分配
   - 否则走分页分配 `alloc_paged_token_slots_extend(...)`
6. L377-L389：调用 `write_cache_indices(...)`
   - 把“prefix + 新分配槽位 out_cache_loc”写入 `req_to_token_pool`
7. L391：返回
   - `out_cache_loc`（本批新 token 的槽位）
   - `req_pool_indices_device` 和 CPU 版本

### `write_cache_indices`（`common.py:78-125`）的作用

- Triton 路径（L91-L107）：批量把 prefix 与新槽位写入 `req_to_token`
- 非 Triton 路径（L109-L124）：Python 循环逐请求写

一句话：它负责维护“逻辑 token 序号 -> 物理 KV 槽位号”的真相表。

---

## 4.2 Decode 阶段：`alloc_for_decode`（`mem_cache/common.py:423-463`）

按代码顺序解释：

1. L431：先做可能的 SWA 回收
2. L435-L450：分配当前步要写的新槽位 `out_cache_loc`
   - 分页时需要先拿到 `last_loc`（上一 token 的物理槽）
   - 计算 `seq_lens_next = seq_lens + token_per_req`
   - 调 `alloc_paged_token_slots_decode`
3. L452-L460：把新分配槽位写进 `req_to_token_pool` 的新位置
4. L462：返回新槽位

---

## 4.3 NPU 分页分配器：`NPUPagedTokenToKVPoolAllocator`

文件：`hardware_backend/npu/allocator_npu.py`

### A) `alloc_extend`（L28-L95）

1. L42-L45：计算每个请求从 prefix 到 seq 需要新增多少 page，并求和
2. L47-L51：空闲页不够则尝试 merge/sort；仍不够返回 `None`
3. L53-L73：小批量时走 NPU kernel `alloc_extend_kernel`
4. L74-L88：大批量走 `alloc_extend_naive`
5. L93：消费掉已分配的 free_pages 头部
6. L94：返回 `out_indices`（即 `out_cache_loc`）

### B) `alloc_decode`（L96-L134）

1. L107-L111：算本轮 decode 需要多少新 page
2. L113-L117：空闲不足则整理 free_pages；还不足返回 `None`
3. L119-L127：关键逻辑
   - `need_new_pages = (seq_lens % page_size == 1)`
   - 若某请求写入后跨页，就从 `free_pages` 取新页起始位置
   - 否则直接 `last_loc + 1`
4. L132：free_pages 前移，表示这些页已占用
5. L133：返回本轮槽位

### C) `free`（L135-L151）

- 把要释放的 token 索引除以 `page_size` 转成 page 索引
- 拼回 `release_pages/free_pages`，供后续复用

---

## 5. KVCache 在前向中的“写入、读取、使用”

核心发生在 `AscendAttnBackend`。

---

## 5.1 Extend（prefill）中 KV 的写读

位置：`ascend_backend.py:747-947`（非 MLA 主干）

关键步骤：

1. L808-L813：选择写入位置
   - 自注意力：`forward_batch.out_cache_loc`
   - 交叉注意力：`forward_batch.encoder_out_cache_loc`
2. L813：`set_kv_buffer(layer, cache_loc, k, v)` 写入 KV
3. L815-L816：`get_key_buffer/get_value_buffer` 读回缓存句柄
4. L887-L899：调用 `_npu_flash_attention_qlens`（满足条件时）
5. L927-L944：否则走 native SDPA 路径

说明：extend 阶段不仅计算当前 token 的注意力结果，也把新 token 的 KV 持久化到池中，供后续 decode 使用。

---

## 5.2 Decode 中 KV 的写读

位置：`ascend_backend.py:1530-1779`

关键步骤（非 MLA）：

1. L1576-L1581：定位 `cache_loc` 并写入 `set_kv_buffer`
2. L1583-L1584：读取整层 `k_cache/v_cache`
3. 三类算子路径：
   - L1608-L1629：`npu_fused_infer_attention_score`（FIA）
   - L1642-L1652：`_npu_paged_attention`（典型 paged decode）
   - L1679-L1694：native SDPA fallback

关键步骤（MLA）：

1. L1698-L1700：写入 `set_kv_buffer(layer, out_cache_loc, k, k_rope)`
2. L1702-L1703：读回 `kv_c/k_pe`
3. L1730-L1747：FIA 路径 `npu_fused_infer_attention_score`
4. L1767-L1777：`_npu_paged_attention_mla`

---

## 5.3 Sparse / TopK 路径

位置：`ascend_backend.py:635-745`

- L659-L661：仍然先写 KV
- L663：读 KV
- L722-L743：`npu_sparse_flash_attention` 做稀疏注意力

---

## 5.4 MLA 预处理路径对 KV 的特殊写法

位置：`hardware_backend/npu/attention/mla_preprocess.py`

关键点：

1. `get_kv_cache_and_cache_idx`（L150-L153）
   - 读出 `(k_cache, v_cache)` + `slot_mapping = out_cache_loc`
2. `forward_absorb_prepare_npu_rms_norm_cache`（L155-L235）
   - L222-L233：调用 `torch.ops.npu.npu_kv_rmsnorm_rope_cache`
   - 该算子在做 RMSNorm+RoPE 的同时把 KV 按 `slotmapping` 写入 cache
3. `forward_mlaprolog(_w8a8)`（L236-L328）
   - `cache_index` 传入 `slot_mapping.to(int64)`
   - `npu_mla_prolog_v3` 在融合算子中完成与 cache 相关处理

---

## 6. ForwardBatch：KV 相关字段在整个执行链的位置

`forward_batch_info.py:273-365` 中，和 KV 最关键的是：

- `out_cache_loc`：本轮输出 token 的物理槽位
- `req_pool_indices`：请求槽位
- `req_to_token_pool`：请求->token 槽位映射表
- `token_to_kv_pool`：真正的 KV 存储池
- `attn_backend`：具体执行 attention（NPU 为 AscendAttnBackend）

这些字段由调度准备阶段灌入，在模型层 attention 里被消费。

---

## 7. 请求处理中 KVCache 的完整时序（把前面串起来）

以一次“普通生成”举例：

1. 请求进 `/generate`，被 tokenizer 处理并发给 scheduler
2. scheduler 将请求放入队列，选入某个 batch
3. 若是 prefill/extend：
   - `alloc_for_extend` 分配 `out_cache_loc`
   - `write_cache_indices` 更新 `req_to_token`
4. 若是 decode：
   - `alloc_for_decode` 为每个请求分配下一个 token 的槽位
   - 也更新 `req_to_token`
5. `ForwardBatch` 带着 `out_cache_loc + kv_pool + req_to_token_pool` 进入模型
6. 每层 `RadixAttention` 调 Ascend backend：
   - 先把当前步 K/V 写入 `token_to_kv_pool`
   - 再从缓存读取历史 K/V，结合 block_table/seq_lens 做注意力
7. 得到 logits 后采样下一 token，scheduler 回传 tokenizer，再回 HTTP 客户端

---

## 8. 关键 NPU 算子与在本流程中的角色

以下按“在本仓库中的调用语义”说明：

1. `torch_npu.npu_scatter_nd_update_`
   - 用途：按槽位索引把 K/V 写进缓存（MHA/MLA 都大量用）
   - 典型位置：`memory_pool_npu.py:142-151, 331-342, 356-362`

2. `torch_npu._npu_reshape_and_cache`
   - 用途：把连续 K/V 重排写入 paged cache（非 FIA 的 MHA 写入路径）
   - 位置：`memory_pool_npu.py:154-164`

3. `torch_npu._npu_flash_attention_qlens`
   - 用途：prefill/extend 场景使用 page table + qlens 做注意力
   - 位置：`ascend_backend.py:887-899`

4. `torch_npu._npu_paged_attention`
   - 用途：decode 阶段单步/小步查询历史 paged KV
   - 位置：`ascend_backend.py:1642-1652`

5. `torch_npu._npu_paged_attention_mla`
   - 用途：MLA 版本的 paged decode 注意力
   - 位置：`ascend_backend.py:1767-1777`

6. `torch.ops.npu.npu_fused_infer_attention_score`
   - 用途：FIA 融合注意力路径（extend/decode 都可能走）
   - 位置：`ascend_backend.py:845-856, 1608-1629, 1730-1747, 1815-1829`

7. `torch_npu.npu_sparse_flash_attention`
   - 用途：TopK 稀疏注意力路径
   - 位置：`ascend_backend.py:722-743`

8. `torch.ops.npu.npu_kv_rmsnorm_rope_cache`
   - 用途：MLA 预处理融合写 cache（RMSNorm + RoPE + cache write）
   - 位置：`mla_preprocess.py:222-233`

9. `torch.ops.custom.npu_mla_prolog_v3`
   - 用途：MLA prolog 融合算子，参数包含 cache 与 cache_index
   - 位置：`mla_preprocess.py:275-277, 315-317`

---

## 9. “读不懂时最容易卡住”的几个点（白话解释）

1. `out_cache_loc` 是什么？  
   - 它是“这次新 token 要写到 KVCache 的物理地址（槽位号）”。

2. `req_to_token_pool` 又是什么？  
   - 它是映射表：`[请求id, token位置] -> 物理槽位号`。  
   - decode 时根据它拿到历史 token 的槽位，再去 KV 池里读。

3. 为什么要 page_size？  
   - 把 KV 按页管理，减少碎片并利于批量管理。  
   - 跨页时才申请新 page，同页内只做 `last_loc + 1`。

4. 为什么看起来“先写再读”？  
   - 对当前步 token，通常先把新 KV 入池，再通过 kernel 按 block_table 读取“上下文 + 当前步”做注意力。

5. MLA 为何有两套缓存（`k_buffer/v_buffer`）且名字看着怪？  
   - MLA 把 no-PE 与 rope 相关部分拆开存，算子再按需要组装；这跟传统 MHA 的 K/V 结构不同。

---

## 10. 代码定位清单（便于继续深挖）

- 请求入口：`python/sglang/srt/entrypoints/http_server.py:669-709`
- tokenizer 发请求：`python/sglang/srt/managers/tokenizer_manager.py:481-529, 1097-1105`
- scheduler 主循环：`python/sglang/srt/managers/scheduler.py:1244-1270`
- 请求建模：`python/sglang/srt/managers/scheduler.py:1611-1845`
- 批次执行：`python/sglang/srt/managers/scheduler.py:2438-2587`
- forward 主分发：`python/sglang/srt/model_executor/model_runner.py:2461-2593`
- attention 入口：`python/sglang/srt/layers/radix_attention.py:99-135`
- KV 分配（extend/decode）：`python/sglang/srt/mem_cache/common.py:328-392, 423-463`
- NPU allocator：`python/sglang/srt/hardware_backend/npu/allocator_npu.py:15-151`
- NPU KV pool：`python/sglang/srt/hardware_backend/npu/memory_pool_npu.py:18-362`
- Ascend attention：`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- MLA 预处理：`python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`

---

## 11. 一句话总结

在 SGLang 的 NPU 路径里，KVCache 的本质是：

- 用 `alloc_for_extend/alloc_for_decode` 先拿到物理槽位（`out_cache_loc`）；
- 用 `req_to_token_pool` 维护请求 token 到物理槽位映射；
- 在 Ascend attention 内通过 `set_kv_buffer` 写入，再通过 paged/sparse/fused 算子读取历史 KV 计算注意力；
- 这样实现高并发请求下可复用、可分页、可回收的 KV 管理。

