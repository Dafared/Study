/**

This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CATLASS_GEMM_KERNEL_FP8_W8A16_MATMUL_TWO_STAGE_HPP
#define CATLASS_GEMM_KERNEL_FP8_W8A16_MATMUL_TWO_STAGE_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class DequantTile_>
class FP8W8A16MatmulTwoStage
{
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using DequantTile = DequantTile_;

    using BlockScheduler = BlockScheduler_;

    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementPrologueB = typename DequantTile::ElementSrc;
    using LayoutPrologueB = typename DequantTile::LayoutSrc;
    using ElementScale = typename DequantTile::ElementScale;
    using LayoutScale = typename DequantTile::LayoutScale;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        typename DequantTile::Params dequantParams;

        GM_ADDR ptrPerGroupScale;
        LayoutScale layoutScale;
        uint32_t groupSize;

        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrPrologueB_,
               LayoutPrologueB const &layoutPrologueB_, GM_ADDR ptrC_, LayoutC const &layoutC_,
               typename DequantTile::Params const &dequantParams_, GM_ADDR ptrPerGroupScale_, 
               LayoutScale layoutScale_, uint32_t groupSize_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrPrologueB(ptrPrologueB_),
              layoutPrologueB(layoutPrologueB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              dequantParams(dequantParams_),
              ptrPerGroupScale(ptrPerGroupScale_),
              layoutScale(layoutScale_),
              groupSize(groupSize_),
              ptrWorkspace(ptrWorkspace_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
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

    static size_t GetWorkspaceSize(Arguments const &args)
    {
        return static_cast<size_t>(args.problemShape.k()) * static_cast<size_t>(args.problemShape.n()) * sizeof(ElementB);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.problemShape,    args.deviceA,     args.layoutA,   args.devicePrologueB,
            args.layoutPrologueB, args.deviceC,     args.layoutC,   {args.deqScalar, args.deqZeroPoint},
            args.deviceScale,     args.layoutScale, args.groupSize, workspace};
        return params;
    }

    CATLASS_DEVICE
    FP8W8A16MatmulTwoStage() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagDequantFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE1>();

        GemmCoord blockShape = L1TileShape::ToCoord();
        BlockScheduler matmulBlockScheduler(params.problemShape, blockShape.GetCoordMN());
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));

        AscendC::GlobalTensor<ElementB> gmDequantB;
        gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace));

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        LayoutB layoutFullB{params.problemShape.k(), params.problemShape.n()};

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);
            GemmCoord offsetCoord = blockIdxCoord * blockShape;

            auto gmBlockA = gmA[params.layoutA.GetOffset(offsetCoord.GetCoordMK())];
            auto layoutBlockA = params.layoutA.GetTileLayout(actualBlockShape.GetCoordMK());

            MatrixCoord offsetB{0, blockIdxCoord.n() * L1TileShape::N};
            auto gmBlockB = gmDequantB[layoutFullB.GetOffset(offsetB)];
            auto layoutBlockB = layoutFullB.GetTileLayout(MatrixCoord{actualBlockShape.k(), actualBlockShape.n()});

            auto gmBlockC = gmC[params.layoutC.GetOffset(offsetCoord.GetCoordMN())];
            auto layoutBlockC = params.layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

            blockMmad(gmBlockA, layoutBlockA, gmBlockB, layoutBlockB, gmBlockC, layoutBlockC, actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t totalAicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        DequantTile dequantTile(resource, params.dequantParams);

        AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
        gmPrologueB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB));

        AscendC::GlobalTensor<ElementB> gmDequantB;
        gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace));

        uint32_t groupSize = params.groupSize;
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

            auto gmBlockScale = gmScaleFull[params.layoutScale.GetOffset(offsetScaleCoord)];
            dequantTile.loadAllTileScales(scaleTileRows, scaleTileCols, scaleStride, gmBlockScale);
            AscendC::PipeBarrier<PIPE_ALL>();

            MatrixCoord offsetCoordB{kOffset, nOffset};
            MatrixCoord actualTileShapeB{kAct, nAct};

            auto gmBlockPrologueB = gmPrologueB[params.layoutPrologueB.GetOffset(offsetCoordB)];
            auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualTileShapeB);

            auto gmBlockDequantB = gmDequantB[layoutFullB.GetOffset(offsetCoordB)];
            auto layoutBlockDequantB = layoutFullB.GetTileLayout(actualTileShapeB);

            auto layoutBlockScale = params.layoutScale.GetTileLayout(scaleTileShape);

            dequantTile(gmBlockDequantB, layoutBlockDequantB, gmBlockPrologueB, layoutBlockPrologueB,
                        layoutBlockScale, groupSize, kOffset, true);
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

#endif  // CATLASS_GEMM_KERNEL_FP8_W8A16_MATMUL_TWO_STAGE_HPP
