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
    using ElementPrologueB = typename BlockMmad::PrologueB::ElementSrc;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutPrologueB = typename BlockMmad::PrologueB::LayoutSrc;
    using LayoutB = typename BlockMmad::LayoutB;

    using ElementScale = typename BlockMmad::PrologueB::ElementScale;
    using LayoutScale = typename BlockMmad::PrologueB::LayoutScale;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using MmadParams = typename BlockMmad::Params;
    using DequantTile = DequantTile_;

    using BlockScheduler = BlockScheduler_;

    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        MmadParams mmadParams;

        GM_ADDR ptrPerGroupScale;
        LayoutScale layoutScale;
        uint32_t groupSize;

        GM_ADDR ptrWorkspace;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrPrologueB_,
               LayoutPrologueB const &layoutPrologueB_, GM_ADDR ptrC_, LayoutC const &layoutC_,
               MmadParams const &mmadParams_, GM_ADDR ptrPerGroupScale_, LayoutScale layoutScale_, uint32_t groupSize_,
               GM_ADDR ptrWorkspace_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrPrologueB(ptrPrologueB_),
              layoutPrologueB(layoutPrologueB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              mmadParams(mmadParams_),
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
            args.layoutPrologueB, args.deviceC,     args.layoutC,   {{}, {args.deqScalar, args.deqZeroPoint}, {}},
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
        auto aicoreNum = AscendC::GetBlockNum();
        auto aicoreIdx = AscendC::GetBlockIdx();

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

        constexpr uint32_t STAGES = 2;
        constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
        constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
        constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
        constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;

        AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
        AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
        AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
        AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
        AscendC::LocalTensor<ElementAccumulator> l0CTensor;

        int32_t l1AEventList[STAGES];
        int32_t l1BEventList[STAGES];
        int32_t l0AEventList[STAGES];
        int32_t l0BEventList[STAGES];

        uint32_t l1AOffset = 0;
        uint32_t l1BOffset = L1A_SIZE * STAGES;

        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
        using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;
        using LayoutAInL1 = layout::zN;
        using LayoutBInL1 = layout::zN;
        using LayoutAInL0 = layout::zZ;
        using LayoutBInL0 = layout::zN;
        using LayoutCInL0 = layout::zN;

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        LayoutB layoutFullB{params.problemShape.k(), params.problemShape.n()};

        uint32_t l1ListId = 0;
        uint32_t l0AListId = 0;
        uint32_t l0BListId = 0;

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);
            GemmCoord offsetCoord = blockIdxCoord * blockShape;

            uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualBlockShape.m());
            uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualBlockShape.n());

            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

            uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualBlockShape.k());

            for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
                uint32_t kActual = (kLoopIdx == kTileCount - 1) ? 
                    (actualBlockShape.k() - kLoopIdx * L1TileShape::K) : L1TileShape::K;

                uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
                uint32_t kActualNext = 0;

                if (kLoopIdx < kTileCount - 1) {
                    uint32_t kLoopIdxNext = kLoopIdx + 1;
                    kActualNext = (kLoopIdxNext < kTileCount - 1) ? L1TileShape::K
                                                                  : (actualBlockShape.k() - kLoopIdxNext * L1TileShape::K);

                    auto l1ATensor = l1ATensorList[l1ListIdNext];
                    auto l1BTensor = l1BTensorList[l1ListIdNext];

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                    auto gmTileA = gmA[params.layoutA.GetOffset(offsetCoord.GetCoordMK()) + params.layoutA.GetOffset(gmTileAOffset)];
                    auto layoutTileA = params.layoutA.GetTileLayout(MakeCoord(actualBlockShape.m(), kActualNext));
                    AscendC::DataCopy(l1ATensor, gmTileA, layoutTileA.Count());
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
                    auto gmTileB = gmDequantB[layoutFullB.GetOffset(gmTileBOffset)];
                    auto layoutTileB = layoutFullB.GetTileLayout(MakeCoord(kActualNext, actualBlockShape.n()));
                    AscendC::DataCopy(l1BTensor, gmTileB, layoutTileB.Count());
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }

                auto l1ATensor = l1ATensorList[l1ListId];
                auto l1BTensor = l1BTensorList[l1ListId];

                uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
                uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);
                uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

                for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                    uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                    for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                        uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                        auto l0ATile = l0ATensorList[l0AListId];
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                        MatrixCoord l1AOffsetCoord{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                        auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffsetCoord)];

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                        if ((mPartIdx == 0) && (kPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                        }
                        AscendC::DataCopy(l0ATile, l1ATile, layoutAInL0.Count());
                        if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                        }

                        for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                            uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                            auto l0BTile = l0BTensorList[l0BListId];
                            LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                            MatrixCoord l1BOffsetCoord{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                            auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffsetCoord)];

                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                            if ((kPartIdx == 0) && (nPartIdx == 0)) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                            }
                            AscendC::DataCopy(l0BTile, l1BTile, layoutBInL0.Count());
                            if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                            MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                            auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                            bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                            uint8_t unitFlag = 0b00;
                            if constexpr (BlockMmad::ENABLE_UNIT_FLAG) {
                                if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                    (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                    unitFlag = 0b11;
                                } else {
                                    unitFlag = 0b10;
                                }
                            }

                            AscendC::MMad<ElementAccumulator, ElementA, ElementB>(
                                l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                            l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                        l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                    }
                }
                l1ListId = l1ListIdNext;
            }

            LayoutC layoutBlock = params.layoutC.GetTileLayout(actualBlockShape.GetCoordMN());
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            auto gmBlockC = gmC[params.layoutC.GetOffset(offsetCoord.GetCoordMN())];
            AscendC::DataCopy(gmBlockC, l0CTensor, layoutBlock.Count());
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t totalAicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        DequantTile dequantTile(resource, params.mmadParams.prologueB);

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

        uint32_t scaleGroupNumk = CeilDiv(kTotal, groupSize);
        uint32_t scaleGroupNumN = CeilDiv(nTotal, groupSize);
        uint32_t scaleStride = params.layoutScale.stride(0);
        dequantTile.loadAllTileScales(scaleGroupNumk, scaleGroupNumN, scaleStride, gmScaleFull);
        AscendC::PipeBarrier<PIPE_ALL>();

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

            MatrixCoord offsetScaleCoord{scaleRowStart, scaleColStart};
            MatrixCoord scaleTileShape{scaleTileRows, scaleTileCols};

            MatrixCoord offsetCoordB{kOffset, nOffset};
            MatrixCoord actualTileShapeB{kAct, nAct};

            auto gmBlockPrologueB = gmPrologueB[params.layoutPrologueB.GetOffset(offsetCoordB)];
            auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualTileShapeB);

            auto gmBlockDequantB = gmDequantB[layoutFullB.GetOffset(offsetCoordB)];
            auto layoutBlockDequantB = layoutFullB.GetTileLayout(actualTileShapeB);

            auto gmBlockScale = gmScaleFull[params.layoutScale.GetOffset(offsetScaleCoord)];
            auto layoutBlockScale = params.layoutScale.GetTileLayout(scaleTileShape);

            dequantTile(gmBlockDequantB, layoutBlockDequantB, gmBlockPrologueB, layoutBlockPrologueB,
                        gmBlockScale, layoutBlockScale, groupSize, kOffset);
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
