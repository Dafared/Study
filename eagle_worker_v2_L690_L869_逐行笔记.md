# EAGLE3（NPU/Ascend）推理执行顺序笔记（全篇顺序重排版）

## 说明
- 本文按**推理时实际执行顺序**组织，不再按“文件来源”散排。
- 重点模型：`EAGLE3 + SGLANG_ENABLE_SPEC_V2=True + NPU(Ascend)`。
- 核心目标：
  - 明确投机推理是在哪里“插入”普通链路的；
  - 明确 prefill / decode 各阶段额外涉及哪些代码；
  - 保留关键函数完整代码块与逐段解释。

## 总览流程（与正文章节对应）
1. **阶段1：启动参数与NPU默认后端落盘**（第1章）  
2. **阶段2：Spec V2 worker 选择**（第2章）  
3. **阶段3：调度器执行点注入投机推理**（第3章）  
4. **阶段4：decode 内存预算与组批逻辑改写**（第4章）  
5. **阶段5：`EAGLEWorkerV2.forward_batch_generation` 主流程**（第5章，逐10行）  
6. **阶段6：verify前准备与NPU分支采样**（第6章）  
7. **阶段7：accepted token 的 logprob 回填**（第7章）  
8. **阶段8：Ascend backend / NPU allocator / NPUGraph 承接**（第8章）  

---

## 1) 启动参数与NPU后端落盘
- 源码定位：
  - `set_default_server_args`: `python/sglang/srt/hardware_backend/npu/utils.py`
  - NPU init 调用: `python/sglang/srt/model_executor/model_runner.py`

### 1.1 NPU默认后端设置（Ascend）

```python
def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for NPU backend.
    """

    # NPU only works with "ascend" attention backend for now
    args.attention_backend = "ascend"
    args.prefill_attention_backend = "ascend"
    args.decode_attention_backend = "ascend"
    if args.page_size is None:
        args.page_size = 128

    # NPU memory settings
    npu_mem = get_npu_memory_capacity()
    if npu_mem <= 32 * 1024:
        # Ascend 910B4,910B4_1
        # (chunked_prefill_size 4k, cuda_graph_max_bs 16 if tp < 4 else 64)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 4 * 1024
        if args.cuda_graph_max_bs is None:
            if args.tp_size < 4:
                args.cuda_graph_max_bs = 16
            else:
                args.cuda_graph_max_bs = 64
    elif npu_mem <= 64 * 1024:
        # Ascend 910B1,910B2,910B2C,910B3,910_9391,910_9392,910_9381,910_9382,910_9372,910_9362
        # (chunked_prefill_size 8k, cuda_graph_max_bs 64 if tp < 4 else 256)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 8 * 1024
        if args.cuda_graph_max_bs is None:
            if args.tp_size < 4:
                args.cuda_graph_max_bs = 64
            else:
                args.cuda_graph_max_bs = 256

    # NPU does not support CustomAllReduce
    args.disable_custom_all_reduce = True

    # handles hierarchical cache configs
    if args.enable_hierarchical_cache:
        args.hicache_io_backend = "kernel_ascend"
        if args.use_mla_backend():
            args.hicache_mem_layout = "page_first_kv_split"
        else:
            args.hicache_mem_layout = "page_first_direct"
```

- 这一步把 prefill/decode attention 统一定向到 `"ascend"`。
- 你的启动参数 `--attention-backend ascend` 与这里保持一致。

### 1.2 NPU backend 初始化触发

```python
if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
```

- 发生在 `ModelRunner` 模块加载阶段。
- 主要完成 NPU runtime 与算子环境初始化。

---

## 2) Spec V2 worker 选择（EAGLE3 -> EAGLEWorkerV2）
- 源码定位：`python/sglang/srt/speculative/spec_info.py`

```python
    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        enable_overlap = not server_args.disable_overlap_schedule
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            # FIXME: migrate to EagleWorker
            if enable_overlap:
                from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                    MultiLayerEagleWorkerV2,
                )

                return MultiLayerEagleWorkerV2

            from sglang.srt.speculative.multi_layer_eagle_worker import (
                MultiLayerEagleWorker,
            )

            return MultiLayerEagleWorker

        elif self.is_eagle():
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
```

- `EAGLE3` 在判定上属于 `is_eagle()`。
- `Spec V2` 的关键条件是 overlap 开启，最终进入 `EAGLEWorkerV2`。

---

## 3) 调度器注入点：投机推理真正“加进去”的地方
- 源码定位：`python/sglang/srt/managers/scheduler.py`

```python
        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # In most cases, we use the model worker batch to run the forward.
                worker_batch_or_batch = batch.get_model_worker_batch()
            else:
                # In speculative decoding v1 (non-overlap) case, we use the batch directly.
                # TODO(lsyin): delete this branch after unifying the abstraction.
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling info will be modified during forward, so we store a copy.
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): move this assignment elsewhere
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): tmp code for spec v2
                    # We only keep future indices for next draft input

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # The future value, usually for next batch preparation
                    # Current implementation strictly synchronizes the seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
```

- **核心注入语句**：`self.model_worker.forward_batch_generation(...)`。
- 非 spec 时 `model_worker = tp_worker`；spec 时 `model_worker = EAGLEWorkerV2`。
- `batch.is_spec_v2` 回写 `next_draft_input`，让下一轮继续 speculative decode。

---

## 4) Decode预算与组批逻辑改写（spec v2）
- 源码定位：`python/sglang/srt/managers/schedule_batch.py`

### 4.1 decode token预算（含 over-allocation）

```python
    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        server_args = get_global_server_args()
        len_per_topk = server_args.speculative_num_steps or 1
        spec_topk = server_args.speculative_eagle_topk or 1
        spec_tokens = server_args.speculative_num_draft_tokens

        if page_size > 1 and spec_topk > 1:
            # last partial page and ceil alignment
            len_per_topk = ceil_align(len_per_topk + page_size, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)
        elif page_size > 1:
            # only page alignment
            len_per_topk = ceil_align(len_per_topk, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)

        num_tokens = max(len_per_topk * spec_topk, spec_tokens) * len(requests)

        # v2 eagle has over-allocation
        return num_tokens * (1 + self.is_spec_v2)
```

- spec v2 会显式放大 decode 内存需求。

### 4.2 decode准备由 spec 路径接管

```python
    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None

        if self.is_spec_v2:
            # TODO(spec-v2): all spec v2 should go through this path
            draft_input: EagleDraftInput = self.spec_info
            draft_input.prepare_for_decode(self)

        if not self.spec_algorithm.is_none():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            return
```

- 这里是普通 decode 与 speculative decode 的分岔点。

---

## 5) `eagle_worker_v2.forward_batch_generation` 逐10行笔记（L690-L869）
- 源码定位：`python/sglang/srt/speculative/eagle_worker_v2.py:L690-L869`

### 5.1 L690-L699

```python
    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
```

- prefill分支先跑 target，并抓 FULL hidden。

### 5.2 L700-L709

```python

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
```

- target prefill 后立刻做 draft prefill。

### 5.3 L710-L719

```python
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.hidden_size,
```

- prefill分支产出 `next_draft_input` 后返回。
- decode分支先确保 `spec_info` 存在。

### 5.4 L720-L729

```python
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
```

- decode第一步：`draft()` 构造 verify 输入。

### 5.5 L730-L739

```python
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output
```

- decode三段式：`draft -> verify -> draft_extend`。

### 5.6 L740-L749

```python

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
```

- verify前做跨流生命周期保护。

### 5.7 L750-L759

```python
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
```

- 在 plan stream 做 target verify 前准备。

### 5.8 L760-L769

```python
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
```

- 主流等待计划流，保证依赖完整。

### 5.9 L770-L779

```python

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
```

- 尝试修正 verify 依赖的 buffer（不同后端实现不同）。

### 5.10 L780-L789

```python
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrive_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrive_next_token.shape
            ).cpu()
```

- grammar 约束需要 CPU 侧辅助结构。

### 5.11 L790-L799

```python

        # Run target verify batch in the main compute stream (GPU compute)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output
```

- target verify 执行，`is_verify=True` 仅返回 logits。

### 5.12 L800-L809

```python

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
```

- 生成结构化输出约束掩码。

### 5.13 L810-L819

```python
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None
```

- 清理旧mask，避免跨轮污染。

### 5.14 L820-L829

```python

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_length

        # Update mamba state for hybrid GDN models after verification
```

- 进入 verify sample，产出 `predict/accept_length/accept_index`。

### 5.15 L830-L839

```python
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_length, accept_index, bs
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()
```

- hybrid_gdn/mamba2 的后处理状态更新。

### 5.16 L840-L849

```python

        if not batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.speculative_num_draft_tokens,
            )
```

- 选每请求“最后接受token”作为下一轮起点 `verified_id`。

### 5.17 L850-L859

```python
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            self._compute_spec_v2_logprobs(batch, logits_output, predict, accept_index)

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
```

- 需要logprob时进入 `_compute_spec_v2_logprobs`。

### 5.18 L860-L869

```python
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )
```

- 返回本轮结果并携带下一轮 draft 输入。

---

## 6) Verify前准备与NPU采样分支
- 归属说明：
  - `EagleVerifyInput` 类定义在 `python/sglang/srt/speculative/eagle_info.py`
  - `prepare_for_v2_verify` / `sample` 的实现来自 `EagleVerifyInputV2Mixin`，定义在 `python/sglang/srt/speculative/eagle_info_v2.py`

### 6.1 `prepare_for_v2_verify`（完整函数）
- 源码定位：`python/sglang/srt/speculative/eagle_info_v2.py`

```python
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            # Set mamba_track_indices for mamba prefix-cache state tracking
            if get_global_server_args().enable_mamba_extra_buffer():
                batch.mamba_track_indices = torch.tensor(
                    [
                        req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                        for req in batch.reqs
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph
```

- 这里会调用 `assign_extend_cache_locs_func`，NPU下走 NPU 专用实现。

### 6.2 `sample`（NPU会走 greedy 分支）
- 源码定位：`python/sglang/srt/speculative/eagle_info_v2.py`

```python
    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_length = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index
```

- 你的 NPU 场景会命中 `or _is_npu`，即 greedy verify 分支。

---

## 7) `_compute_spec_v2_logprobs` 全函数逐10行笔记
- 源码定位：`python/sglang/srt/speculative/eagle_worker_v2.py:L871-L936`

### 7.1 L871-L880

```python
    def _compute_spec_v2_logprobs(
        self,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
    ):
        """Compute logprobs for accepted tokens on GPU in the forward stream.

        Stores results in logits_output fields so they flow through copy_to_cpu().
```

### 7.2 L881-L890

```python
        """

        bs = len(batch.seq_lens)
        max_accept = self.speculative_num_steps + 1
        device = predict.device

        flat_accept_idx = accept_index.long().reshape(-1)
        gathered_logits = logits_output.next_token_logits[flat_accept_idx]

        if (
```

### 7.3 L891-L900

```python
            batch.sampling_info.is_all_greedy
            or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
        ):
            gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
        else:
            temperatures = torch.repeat_interleave(
                batch.sampling_info.temperatures,
                max_accept,
                dim=0,
            )
```

### 7.4 L901-L910

```python
            gathered_logprobs = torch.nn.functional.log_softmax(
                gathered_logits / temperatures, dim=-1
            )
        gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

        accepted_token_ids = predict[flat_accept_idx]
        token_logprobs = gathered_logprobs[
            torch.arange(bs * max_accept, device=device),
            accepted_token_ids.long(),
        ]
```

### 7.5 L911-L920

```python
        logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

        if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
            top_logprobs_nums_expanded = [
                num for num in batch.top_logprobs_nums for _ in range(max_accept)
            ]
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
```

### 7.6 L921-L930

```python
                gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
            )

        if batch.token_ids_logprobs and any(
            x is not None for x in batch.token_ids_logprobs
        ):
            token_ids_logprobs_expanded = [
                ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
            ]
            (
```

### 7.7 L931-L936

```python
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
            )
```

- 仅针对 accepted positions 计算 logprob，再回写到 `logits_output`。

---

## 8) Ascend后端承接：attention / allocator / graph
- 源码定位：
  - attention 注册：`python/sglang/srt/layers/attention/attention_registry.py`
  - attention 选择：`python/sglang/srt/model_executor/model_runner.py`
  - Ascend metadata：`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
  - NPU allocator/KV：`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  - NPU graph replay：`python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`

### 8.1 attention backend 注册与选择

```python
@register_attention_backend("ascend")
def create_ascend_backend(runner):
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    return AscendAttnBackend(runner)
```

```python
    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )
        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()
        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )
        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend
```

- prefill/decode 的 backend 最终在这里决议。

### 8.2 NPU专用 KV pool / allocator

```python
        # Initialize token_to_kv_pool
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        if self.server_args.attention_backend == "ascend" and not self.mambaish_config:
            if self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )
                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_nsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )
                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
```

```python
        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
                from sglang.srt.hardware_backend.npu.allocator_npu import (
                    NPUPagedTokenToKVPoolAllocator,
                )
                self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
```

- speculative decode 的 over-allocation、retract、verify 都直接作用在这一层分配器。

### 8.3 Ascend verify metadata 差异

```python
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens = forward_batch.extend_seq_lens
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        if forward_batch.seq_lens is not None:
            self.forward_metadata.seq_lens = forward_batch.seq_lens.int()
        else:
            self.forward_metadata.seq_lens = forward_batch.seq_lens_cpu.to(
                self.device
            ).int()

        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens
```

- `target_verify` 在 Ascend 后端会显式加上 speculative draft token 维度。

### 8.4 NPUGraph replay 中 target_verify 分支

```python
    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)

        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        # Replay
        if not is_deepseek_nsa(self.model_runner.model_config.hf_config):
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.num_tokens_per_bs
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    self.bs - self.raw_bs
                )
```

- target_verify 的 seq_lens 处理与普通路径不同，是 NPU图执行关键差异。

---

## 结论：投机推理在代码中的“新增层次”
- **调度层新增**：`scheduler.run_batch` 通过 `model_worker` 切入 speculative worker。
- **批处理层新增**：`ScheduleBatch` 在 decode预算/prepare 上引入 spec v2 规则。
- **推理层新增**：`EAGLEWorkerV2` 把 decode改写为 `draft -> verify -> draft_extend`。
- **验证层新增**：`verify_input.sample`（NPU走greedy）+ `_compute_spec_v2_logprobs`（accepted回填）。
- **后端层适配**：Ascend attention metadata、NPU allocator、NPUGraph replay 对 target_verify 特殊处理。

## 追踪清单（从入口到后端）
- `server_args / npu.utils`：参数与后端初始化
- `spec_info.create_worker`：worker路由
- `scheduler.run_batch`：注入投机推理
- `schedule_batch`：decode预算与prepare
- `eagle_worker_v2`：prefill/decode主逻辑
- `eagle_info`：`EagleVerifyInput/EagleDraftInput` 类定义
- `eagle_info_v2`：verify准备与sample（mixin实现）
- `tp_worker`：target verify前向分支
- `model_runner / ascend_backend / npu_graph_runner / kv_cache_mixin`：后端承接
