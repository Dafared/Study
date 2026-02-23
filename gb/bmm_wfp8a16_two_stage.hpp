/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_TWO_STAGE_HPP
#define CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_TWO_STAGE_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class DequantTile_>
class BatchMatmulFP8TwoStage
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementPrologueB = typename DequantTile_::ElementSrc;
    using LayoutPrologueB = typename DequantTile_::LayoutSrc;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using DequantTile = DequantTile_;
    using ElementScale = typename DequantTile::ElementScale;
    using LayoutScale = typename DequantTile::LayoutScale;
    using BlockScheduler = BlockScheduler_;

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        int64_t strideB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;

        typename DequantTile::Params dequantParams;

        GM_ADDR ptrPerGroupScale;
        LayoutScale layoutScale;
        int64_t strideS;
        uint32_t groupSize;

        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrA_, LayoutA const &layoutA_,
               int64_t strideA_, GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_, int64_t strideB_,
               GM_ADDR ptrC_, LayoutC const &layoutC_, int64_t strideC_,
               typename DequantTile::Params const &dequantParams_, GM_ADDR ptrPerGroupScale_,
               LayoutScale layoutScale_, int64_t strideS_, uint32_t groupSize_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              strideA(strideA_),
              ptrPrologueB(ptrPrologueB_),
              layoutPrologueB(layoutPrologueB_),
              strideB(strideB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              strideC(strideC_),
              dequantParams(dequantParams_),
              ptrPerGroupScale(ptrPerGroupScale_),
              layoutScale(layoutScale_),
              strideS(strideS_),
              groupSize(groupSize_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR deviceA;
        LayoutA layoutA;
        GM_ADDR devicePrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR deviceC;
        LayoutC layoutC;
        half deqScalar;
        half deqZeroPoint;
        GM_ADDR deviceScale;
        LayoutScale layoutScale;
        uint32_t groupSize;
        uint32_t aicoreNum;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return static_cast<size_t>(args.problemCount) * static_cast<size_t>(args.problemShape.k()) *
               static_cast<size_t>(args.problemShape.n()) * sizeof(ElementB);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        int64_t strideS = ((problemShape.k() + 127) / args.groupSize) * ((problemShape.n() + 127) / args.groupSize);
        Params params{args.problemShape,    args.problemCount, args.deviceA,
                     args.layoutA,         strideA,           args.devicePrologueB,
                     args.layoutPrologueB, strideB,           args.deviceC,
                     args.layoutC,         strideC,           {args.deqScalar, args.deqZeroPoint},
                     args.deviceScale,     args.layoutScale,  strideS,
                     args.groupSize,       workspace};
        return params;
    }

    CATLASS_DEVICE
    BatchMatmulFP8TwoStage() {}

    CATLASS_DEVICE
    ~BatchMatmulFP8TwoStage() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagDequantFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE1>();

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = params.problemCount * matmulBlockScheduler.GetCoreLoops();
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            int64_t batchOffsetA = batchIdx * params.strideA;
            int64_t batchOffsetC = batchIdx * params.strideC;
            int64_t batchOffsetB = batchIdx * params.strideB;

            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            AscendC::GlobalTensor<ElementB> gmDequantB;
            gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + batchOffsetB);

            LayoutB layoutFullB{params.problemShape.k(), params.problemShape.n()};
            MatrixCoord offsetB{0, blockCoord.n() * L1TileShape::N};
            auto gmBlockB = gmDequantB[layoutFullB.GetOffset(offsetB)];
            auto layoutBlockB = layoutFullB.GetTileLayout(MatrixCoord{actualBlockShape.k(), actualBlockShape.n()});

            blockMmad(gmA[batchOffsetA + gmOffsetA], params.layoutA, gmBlockB, layoutBlockB,
                      gmC[batchOffsetC + gmOffsetC], params.layoutC, actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        DequantTile dequantTile(resource, params.dequantParams);

        uint32_t totalAicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        uint32_t groupSize = params.groupSize;
        AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
        gmPrologueB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB));

        AscendC::GlobalTensor<ElementScale> gmScaleFull;
        gmScaleFull.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrPerGroupScale));

        uint32_t kTotal = params.problemShape.k();
        uint32_t nTotal = params.problemShape.n();
        uint32_t mTotal = params.problemShape.m();

        GemmCoord dequantProblemShape{kTotal, nTotal, 0};
        MatrixCoord dequantTileMN{L1TileShape::K, L1TileShape::N};

        LayoutB layoutFullB{kTotal, nTotal};

        constexpr uint32_t SWIZZLE_OFFSET = 3;
        uint32_t swizzleDirection = (mTotal > nTotal) ? 0 : 1;

        Block::DynamicGemmIdentityBlockSwizzle dequantBlockScheduler(
            dequantProblemShape, dequantTileMN, SWIZZLE_OFFSET, swizzleDirection);
        uint32_t dequantLoops = dequantBlockScheduler.GetCoreLoops();

        for (uint32_t batchIdx = 0; batchIdx < params.problemCount; ++batchIdx) {
            int64_t batchOffsetB = batchIdx * params.strideB;
            int64_t batchOffsetS = batchIdx * params.strideS;

            AscendC::GlobalTensor<ElementPrologueB> gmPrologueBBatch;
            gmPrologueBBatch.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB + batchOffsetB));

            AscendC::GlobalTensor<ElementB> gmDequantB;
            gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + batchOffsetB);

            AscendC::GlobalTensor<ElementScale> gmScaleBatch;
            gmScaleBatch.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrPerGroupScale +
                                                                                batchOffsetS * sizeof(ElementScale)));

            for (uint32_t loopIdx = aicoreIdx; loopIdx < dequantLoops; loopIdx += totalAicoreNum) {
                auto blockIdxCoord = dequantBlockScheduler.GetBlockCoord(loopIdx);

                uint32_t kTileIdx = blockIdxCoord.m();
                uint32_t nTileIdx = blockIdxCoord.n();

                uint32_t kOffset = kTileIdx * L1TileShape::K;
                uint32_t nOffset = nTileIdx * L1TileShape::N;

                uint32_t kAct = (kOffset + L1TileShape::K <= kTotal) ? L1TileShape::K : (kTotal - kOffset);
                uint32_t nAct = (nOffset + L1TileShape::N <= nTotal) ? L1TileShape::N : (nTotal - nOffset);

                uint32_t scaleRowStart = kOffset / groupSize;
                uint32_t scaleRowEnd = (kOffset + kAct - 1) / groupSize;
                uint32_t scaleColStart = nOffset / groupSize;
                uint32_t scaleColEnd = (nOffset + nAct - 1) / groupSize;

                uint32_t scaleTileRows = scaleRowEnd - scaleRowStart + 1;
                uint32_t scaleTileCols = scaleColEnd - scaleColStart + 1;

                uint32_t scaleStride = params.layoutScale.stride(0);

                MatrixCoord offsetScaleCoord{scaleRowStart, scaleColStart};
                MatrixCoord scaleTileShape{scaleTileRows, scaleTileCols};

                auto gmBlockScale = gmScaleBatch[params.layoutScale.GetOffset(offsetScaleCoord)];
                dequantTile.loadAllTileScales(scaleTileRows, scaleTileCols, scaleStride, gmBlockScale);
                AscendC::PipeBarrier<PIPE_ALL>();

                MatrixCoord offsetCoordB{kOffset, nOffset};
                MatrixCoord actualTileShapeB{kAct, nAct};

                auto gmBlockPrologueB = gmPrologueBBatch[params.layoutPrologueB.GetOffset(offsetCoordB)];
                auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualTileShapeB);

                auto gmBlockDequantB = gmDequantB[layoutFullB.GetOffset(offsetCoordB)];
                auto layoutBlockDequantB = layoutFullB.GetTileLayout(actualTileShapeB);

                auto layoutBlockScale = params.layoutScale.GetTileLayout(scaleTileShape);

                dequantTile(gmBlockDequantB, layoutBlockDequantB, gmBlockPrologueB, layoutBlockPrologueB,
                            layoutBlockScale, groupSize, kOffset, true);
            }
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagDequantFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_DEQUANT_FINISH = 0x2;
    Arch::CrossCoreFlag flagDequantFinish{FLAG_DEQUANT_FINISH};
    Arch::Resource<ArchTag> resource;
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_BATCH_MATMUL_M_FP8_TWO_STAGE_HPP
