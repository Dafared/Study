/**

This program is free software, you can redistribute it and/or modify.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_TWO_STAGE_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_TWO_STAGE_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_, class DequantTile_>
class GroupedMatmulSliceMFP8W8A16TwoStage
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
    using ElementGroupList = ElementGroupList_;
    using BlockScheduler = BlockScheduler_;

    struct Params {
        GemmCoord problemShape;
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
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
        Params(GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_, GM_ADDR ptrA_,
               LayoutA const &layoutA_, GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_, GM_ADDR ptrC_,
               LayoutC const &layoutC_, typename DequantTile::Params const &dequantParams_, GM_ADDR ptrPerGroupScale_,
               LayoutScale layoutScale_, uint32_t groupSize_, GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              problemCount(problemCount_),
              ptrGroupList(ptrGroupList_),
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
        uint32_t problemCount;
        GM_ADDR ptrGroupList;
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
        Params params{args.problemShape,    args.problemCount, args.ptrGroupList,    args.deviceA,
                     args.layoutA,         args.devicePrologueB, args.layoutPrologueB, args.deviceC,
                     args.layoutC,         {args.deqScalar, args.deqZeroPoint},     args.deviceScale,
                     args.layoutScale,     args.groupSize,       workspace};
        return params;
    }

    CATLASS_DEVICE
    GroupedMatmulSliceMFP8W8A16TwoStage() {}

    CATLASS_DEVICE
    ~GroupedMatmulSliceMFP8W8A16TwoStage() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagDequantFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE1>();

        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetC = 0;
        int64_t gmGroupOffsetB = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx)
                                                : (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutFullB{inGroupProblemShape.k(), inGroupProblemShape.n()};
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmDequantB;
            gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmGroupOffsetB);

            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                MatrixCoord offsetB{0, blockCoord.n() * L1TileShape::N};
                auto gmBlockB = gmDequantB[layoutFullB.GetOffset(offsetB)];
                auto layoutBlockB = layoutFullB.GetTileLayout(MatrixCoord{actualBlockShape.k(), actualBlockShape.n()});

                blockMmad(gmA[gmGroupOffsetA + gmOffsetA], layoutA, gmBlockB, layoutBlockB,
                          gmC[gmGroupOffsetC + gmOffsetC], layoutC, actualBlockShape);
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        DequantTile dequantTile(resource, params.dequantParams);

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(reinterpret_cast<__gm__ ElementGroupList *>(params.ptrGroupList));

        uint32_t totalAicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

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

        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetS = 0;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
            gmPrologueB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB + gmGroupOffsetB));

            AscendC::GlobalTensor<ElementB> gmDequantB;
            gmDequantB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmGroupOffsetB);

            AscendC::GlobalTensor<ElementScale> gmScaleFullGroup;
            gmScaleFullGroup.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.ptrPerGroupScale +
                                                                                 gmGroupOffsetS * sizeof(ElementScale)));

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

                auto gmBlockScale = gmScaleFullGroup[params.layoutScale.GetOffset(offsetScaleCoord)];
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

            gmGroupOffsetB += kTotal * nTotal;
            gmGroupOffsetS += ((kTotal + 127) / groupSize) * ((nTotal + 127) / groupSize);
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

#endif  // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_FP8_W8A16_TWO_STAGE_HPP
