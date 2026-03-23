# Soft FP8 Matmul 优化方案设计：K-Tiling 与 硬件级 Tail Split-K 融合

## 1. 核心痛点与优化目标

在 Ascend 910B/C 硬件上，针对大 M 场景（如 Prefill 阶段的 `m >= 256` 且 `n > aicoreNum * 128`），当前代码采用了一种 **N-only 分核逻辑**：让每个核包揽一个 N 块（宽度 128）的完整 K 维度的反量化与计算，从而消除 Swizzle 调度中多个核对同一块权重的重复反量化。

然而，这种调度在 LLM 推理场景下暴露了两个致命的短板：

1. **OOM 风险 (Workspace 爆炸)**：
   AIV 会一次性将长度为 $K$ 的 FP8 权重反量化为 BF16 并存入 Workspace。对于 LLaMA-3 的 FFN 层（$K \approx 28672$），假设 NPU 有 40 个核，Workspace 会瞬间飙升至 $\sim 300\text{MB}$。在显存被 KV Cache 大量占用的 LLM 推理中，这极易引发 OOM。
2. **尾部效应 (算力闲置)**：
   当 N 维度的块数不能被物理核数（`aicoreNum`）整除时，最后一轮循环只有部分核在工作，其余核处于 idle 状态。例如 `N = 3000` 需 24 个块，第二轮仅 4 个核在扛巨大的 $M \times K$ 计算，成为性能瓶颈。

**优化目标**：
在**不增加冗余反量化**的前提下，同时解决“显存占用过大”和“尾部算力闲置”问题，并且**杜绝引入软件级的 Reduce 开销**。

---

## 2. 优化架构设计：K-Tiling + 硬件级 Atomic Add

我们提出一套融合的“双层切分”调度架构。核心思想是：利用硬件 MTE3 引擎的 `SetAtomicAdd` 特性，在写回 Global Memory (GM) 时进行无锁累加，从而彻底抛弃中间暂存（Partial Workspace）和软件归约。

### 2.1 第一层：全局 K-Chunk (解决 OOM)
我们不再一次性反量化完整的 $K$，而是引入 `K_CHUNK`（例如 2048）。
* **AIV 行为**：每次只反量化 `K_CHUNK * 128` 的权重。
* **AIC 行为**：复用这块权重计算所有的 $M$，得出部分 $C$。
* **合并机制**：当计算第一个 Chunk 时，直接覆盖写入 GM；计算后续 Chunk 时，开启 `SetAtomicAdd`，由硬件自动累加到最终的 $C$ 矩阵上。
* **收益**：Workspace 占用从 $O(K)$ 断崖式下降为 $O(K\_CHUNK)$（以 40 核为例，仅需 20MB 的 Double Buffer 空间）。

### 2.2 第二层：Tail Split-K (解决尾部效应)
在 N-Tile 循环到达最后一轮（尾部）时，我们利用已经建立的 `SetAtomicAdd` 累加机制，顺水推舟地进行 K 维度的打散。
* **动态切刀**：剩余的尾部块数为 `tailTiles = nTileCount % aicoreNum`。闲置核数为 `aicoreNum - tailTiles`。我们将当前的 $K$ 维度（或 $K\_CHUNK$）切分为 `kSplitNum = aicoreNum / tailTiles` 份。
* **任务分发**：原本由 1 个核承担的尾部 N-Tile 计算，现在由 `kSplitNum` 个核共同承担（每个核计算不同的 K 碎片）。
* **零开销归约**：因为这些核都在算同一个 $C$ 矩阵的不同 K 碎片，我们强制开启 `SetAtomicAdd`。所有核算完后各自通过 Fixpipe 扔给 GM，硬件保证累加正确。**零中间内存，零软件 Reduce 耗时。**

---

## 3. 详细代码修改方案

### 3.1 Host 侧：Workspace 动态缩减
*文件参考：`csrc/catlass/op_host/catlass_matmul_fp8.cpp`*

不再按照完整的 $K$ 分配，而是基于静态的 `K_CHUNK` 分配 Double Buffer 空间。

```cpp
// 新增 K_CHUNK 宏或常量
constexpr uint32_t K_CHUNK = 2048; 

// Workspace 缩减为 2 倍的 K_CHUNK (用于 PingPong)
uint64_t workspace_size_n_only = 2ULL * K_CHUNK * 128 * sizeof(bfloat16_t) * aicCoreNum;

bool use_n_only = (m > 256) && (n > aicCoreNum * 128);
```

### 3.2 Kernel 侧：双层循环与 AtomicAdd
*文件参考：`csrc/catlass/utils/gemm/kernel/mm_wfp8a16.hpp` -> `FP8W8A16MatmulNOnly::operator()<AIC>`*

以下是 AIC 侧的控制流重构（AIV 侧对应增加 K_CHUNK 外循环，反量化时传入 `kSplitOffset` 即可）。

```cpp
template <>
CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
{
    auto aicoreNum = AscendC::GetBlockNum();
    auto aicoreIdx = AscendC::GetBlockIdx();
    BlockMmad blockMmad(resource);

    uint32_t nTileCount = CeilDiv(params.problemShape.n(), L1TileShape::N);
    uint32_t fullTaskEnd = (nTileCount / aicoreNum) * aicoreNum;
    uint32_t tailTiles = nTileCount % aicoreNum;

    constexpr uint32_t K_CHUNK = 2048;
    uint32_t kChunkCount = CeilDiv(params.problemShape.k(), K_CHUNK);

    // 1. 最外层 K-Tiling 循环 (防 OOM)
    for (uint32_t kChunkIdx = 0; kChunkIdx < kChunkCount; ++kChunkIdx) {
        uint32_t kOffset = kChunkIdx * K_CHUNK;
        uint32_t kAct = min(K_CHUNK, params.problemShape.k() - kOffset);
        
        bool isFirstChunk = (kChunkIdx == 0);

        // 2. 区分满载阶段与尾部阶段
        if (nTileIdx < fullTaskEnd) {
            // ==========================================
            // Phase 1: 满载阶段 (正常的 N-Tile 计算)
            // ==========================================
            for (uint32_t nTileIdx = aicoreIdx; nTileIdx < fullTaskEnd; nTileIdx += aicoreNum) {
                Catlass::Arch::CrossCoreWaitFlag(flagDequantReady[pingpong]);
                
                uint32_t nOffset = nTileIdx * L1TileShape::N;
                uint32_t nAct = min(L1TileShape::N, params.problemShape.n() - nOffset);
                LayoutB layoutBlockB{params.problemShape.k(), nAct};

                // 设置硬件原子累加
                if (!isFirstChunk) AscendC::SetAtomicAdd<ElementC>();
                else AscendC::SetAtomicNone();

                for (uint32_t mOffset = 0; mOffset < params.problemShape.m(); mOffset += L1TileShape::M) {
                    uint32_t mAct = min(L1TileShape::M, params.problemShape.m() - mOffset);
                    GemmCoord actualBlockShape{mAct, nAct, kAct};
                    
                    // A 矩阵的 K 偏移
                    GemmCoord offsetCoordA{mOffset, kOffset, 0}; 
                    GemmCoord offsetCoordC{mOffset, nOffset, 0};

                    auto gmBlockA = gmA[params.layoutA.GetOffset(offsetCoordA.GetCoordMK())];
                    auto gmBlockC = gmC[params.layoutC.GetOffset(offsetCoordC.GetCoordMN())];

                    // 执行 Mmad (内部 Fixpipe 时会自动应用上面的 Atomic 配置)
                    blockMmad(gmBlockA, ..., gmBlockC, ...);
                }
                
                AscendC::SetAtomicNone(); // 恢复默认状态
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE2>(flagDequantConsumed[pingpong]);
            }
        } else if (tailTiles > 0) {
            // ==========================================
            // Phase 2: 尾部阶段 (Tail Split-K，防算力闲置)
            // ==========================================
            // 计算 K 的切分份数 (向下取整保证不超核)
            uint32_t kSplitNum = aicoreNum / tailTiles; 
            
            if (kSplitNum > 1) {
                // 强制对齐到 groupSize(128)，保证 AIV Scale 寻址不跨块
                uint32_t kSplitSize = RoundUp(CeilDiv(kAct, kSplitNum), params.groupSize);
                uint32_t tailTotalTasks = tailTiles * kSplitNum;

                if (aicoreIdx < tailTotalTasks) {
                    uint32_t tailTileOffset = aicoreIdx % tailTiles;
                    uint32_t kSplitIdx = aicoreIdx / tailTiles;
                    
                    uint32_t nTileIdx = fullTaskEnd + tailTileOffset;
                    uint32_t nOffset = nTileIdx * L1TileShape::N;
                    
                    uint32_t kSplitOffset = kOffset + kSplitIdx * kSplitSize;
                    uint32_t kSplitAct = min(kSplitSize, params.problemShape.k() - kSplitOffset);

                    Catlass::Arch::CrossCoreWaitFlag(flagDequantReady[pingpong]);

                    // 【核心差异】在 Tail Split-K 下，多个核写同一个 C 块，
                    // 无论是不是第一个 Chunk，都必须强制开启 AtomicAdd 避免覆盖！
                    AscendC::SetAtomicAdd<ElementC>();

                    for (uint32_t mOffset = 0; mOffset < params.problemShape.m(); mOffset += L1TileShape::M) {
                        // 逻辑同上，传入 kSplitOffset 和 kSplitAct
                        // ...
                        blockMmad(gmBlockA, ..., gmBlockC, ...);
                    }
                    
                    AscendC::SetAtomicNone();
                    Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE2>(flagDequantConsumed[pingpong]);
                }
            } else {
                // 闲置核极少，直接硬扛 (类似 Phase 1 逻辑)
            }
        }
        pingpong = 1 - pingpong;
    }
}
```

---

## 4. 边界条件与 API 正确性校验

1. **SetAtomicAdd 的执行归属**：
   - 依据 Ascend 架构，`SetAtomicAdd` 属于 MTE3（数据搬运单元）的寄存器配置。
   - 在分离架构下，MTE3 属于 AIC（Cube 核）的流水线。因此在 `operator()<AscendC::AIC>` 中调用合法，且**完全不需要 AIV 介入**。参考代码：`catlass/include/catlass/gemm/block/block_mmad_single_core_splitk.hpp:218`。
2. **Scale 索引的安全性**：
   - AIV 在读取 Scale 时，是通过 `scaleRowStart = kOffset / groupSize` 计算的。
   - 我们的设计中通过 `RoundUp(..., params.groupSize)` 强制 `kSplitSize` 向上对齐到了 128。这保证了 `kOffset` 和 `kSplitOffset` 永远是 128 的整数倍，避免了 Scale block 被切碎引发的越界错误。
3. **Workspace 的 Double Buffer 安全性**：
   - AIV 每次写入 Workspace 前会执行 `CrossCoreWaitFlag(flagDequantConsumed)`。只要 AIC 的内层 M 循环彻底退出并 Set Flag，Workspace 就可以安全复用，不会发生读写踩踏。
4. **性能约束**：
   - 开启 AtomicAdd 意味着写入 L2 Cache 时触发 Read-Modify-Write。但由于我们在最内层复用了 $M$ 和 $K\_CHUNK$ 的计算，这种写回的频率非常低，相比于消除的 128MB Partial C 读写以及 AIV 唤醒开销，性能收益是压倒性的。