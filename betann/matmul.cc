#include "betann/matmul.h"

#include <fmt/format.h>

#include "betann/kernels.h"
#include "betann/kernels_helper.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

std::tuple<bool /* transposed */, bool /* needsCopy */, uint32_t /* stride */>
NeedsContiguousCopy(const std::vector<uint32_t>& shape,
                    const std::vector<uint32_t>& strides,
                    bool isVector) {
  auto stx = strides[strides.size() - 2];
  auto sty = strides[strides.size() - 1];
  if (sty == 1 && (!isVector || stx == shape[shape.size() - 1]))
    return {false, false, stx};
  if (stx == 1 && (!isVector || sty == shape[shape.size() - 2]))
    return {true, false, sty};
  return {false, true, shape[shape.size() - 1]};
}

Buffer CopyArray(Device& device,
                 DataType dataType,
                 const Buffer& src,
                 const std::vector<uint32_t>& shape,
                 const std::vector<uint32_t>& strides) {
  Buffer dst = device.CreateBuffer(
      NumElements(shape) * SizeOf(dataType),
      BufferUsage::Storage | BufferUsage::CopyDst);
  CopyGeneral(device, dataType, dst, dataType, src, shape, strides);
  return dst;
}

}  // namespace

void MatrixVectorMultiply(Device& device,
                          DataType dataType,
                          const std::vector<uint32_t>& batchShape,
                          const Buffer& out,
                          const Buffer& mat,
                          bool matTranspose,
                          uint32_t matRows,
                          uint32_t matCols,
                          uint32_t matRowStride,
                          const std::vector<uint32_t>& batchStridesMat,
                          const Buffer& vec,
                          const std::vector<uint32_t>& batchStridesVec,
                          bool disableSubgroups) {
#ifndef __APPLE__
  // There is no way to control subgroup size and it is usually too small for
  // gemvt kernel.
  if (matTranspose)
    disableSubgroups = true;
#endif

  // Determine the parameters according to data size.
  uint32_t groupCount, groupRows, groupCols, rowWorkPerThread, colWorkPerThread;
  if (matTranspose) {
    if (matCols >= 2048)
      groupCount = 16;
    else if (matCols >= 512)
      groupCount = 4;
    else
      groupCount = 2;
    groupRows = 8;
    groupCols = 4;
    rowWorkPerThread = 4;
    colWorkPerThread = matCols < 4 ? 1 : 4;
  } else {
    groupCount = matRows >= 4096 ? 8 : 4;
    groupRows = 1;  // not used in gemv
    groupCols = 32;  // not used in gemv
    rowWorkPerThread = matRows < 4 ? 1 : 4;
    colWorkPerThread = 4;
  }

  bool contiguous = batchShape.size() < 2;
  bool enableF16 = EnableF16(device, dataType);
  auto capacities = GetCapacityVariables(device, enableF16, disableSubgroups);
  RunKernel(device,
            matTranspose ? "gemvt" : "gemv",
            fmt::format("gemv_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                        matTranspose,
                        contiguous,
                        std::get<bool>(capacities["enable_subgroups"]),
                        WgslType(dataType),
                        groupCount,
                        groupRows,
                        groupCols,
                        rowWorkPerThread,
                        colWorkPerThread),
            [&]() {
              return Append(
                  ParseTemplate(
                      matTranspose ? wgsl_source_gemvt : wgsl_source_gemv,
                      {
                        {"contiguous", contiguous},
                        {"dtype", WgslType(dataType)},
                        {"dtype_is_floating", IsFloating(dataType)},
                        {"group_count", groupCount},
                        {"group_rows", groupRows},
                        {"group_cols", groupCols},
                        {"row_work_per_thread", rowWorkPerThread},
                        {"col_work_per_thread", colWorkPerThread},
                      },
                      capacities),
                  wgsl_source_utils);
            },
            {
              out,
              mat,
              device.CreateBufferFromScalar(matRows),
              device.CreateBufferFromScalar(matCols),
              device.CreateBufferFromScalar(matRowStride),
              batchStridesMat.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(batchStridesMat),
              vec,
              batchStridesVec.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(batchStridesVec),
              !contiguous ? device.CreateBufferFromVector(batchShape) : nullptr,
            },
            {
              matTranspose
                  ? DivCeil(matCols, colWorkPerThread * groupCount * groupCols)
                  : DivCeil(matRows, rowWorkPerThread * groupCount * groupRows),
              1,
              NumElements(batchShape),
            });
}

void MatrixMultiply(Device& device,
                    DataType dataType,
                    const Buffer& out,
                    Buffer a,
                    const std::vector<uint32_t>& aShape,
                    const std::vector<uint32_t>& aStrides,
                    Buffer b,
                    const std::vector<uint32_t>& bShape,
                    const std::vector<uint32_t>& bStrides) {
  if (aShape.size() < 2 || bShape.size() < 2)
    throw std::runtime_error("Inputs of MatrixMultipy must be matrices.");

  // Return 0s if either input is empty.
  if (a.GetSize() == 0 || b.GetSize() == 0) {
    CopyContiguous(device, CopyType::Scalar,
                   dataType, out, out.GetSize() / SizeOf(dataType),
                   DataType::U32, device.CreateBufferFromScalar(0u));
    return;
  }

  uint32_t M = aShape[aShape.size() - 2];
  uint32_t K = aShape[aShape.size() - 1];
  uint32_t N = bShape[bShape.size() - 1];
  bool isMatrixVector = M == 1 || N == 1;

  // Check transpose state and do contiguous copies when necessary.
  auto [aTransposed, aNeedsCopy, aLeadingStride] =
      NeedsContiguousCopy(aShape, aStrides, M == 1);
  auto [bTransposed, bNeedsCopy, bLeadingStride] =
      NeedsContiguousCopy(bShape, bStrides, N == 1);
  if (aNeedsCopy)
    a = CopyArray(device, dataType, a, aShape, aStrides);
  if (bNeedsCopy)
    b = CopyArray(device, dataType, b, bShape, bStrides);

  // Collapse batch dimensions.
  std::vector<uint32_t> aBatchShape = Slice(aShape, 0, -2);
  std::vector<uint32_t> bBatchShape = Slice(bShape, 0, -2);
  if (aBatchShape != bBatchShape)
    throw std::runtime_error("Matrices have incorrectly broadcasted shapes.");
  auto [batchShape, aBatchStrides, bBatchStrides] =
      CollapseContiguousDims(aBatchShape,
                             Slice(aStrides, 0, -2),
                             Slice(bStrides, 0, -2));
  if (batchShape.empty()) {
    batchShape = {1};
    aBatchStrides = bBatchStrides = {0};
  }

  // Collapse batches into M if possible.
  if (batchShape.size() == 1 &&
      aStrides[aStrides.size() - 2] == K &&
      aBatchStrides[aBatchStrides.size() - 1] == M * K &&
      bBatchStrides[bBatchStrides.size() - 1] == 0 &&
      NumElements(batchShape) > 1) {
    M *= batchShape[0];
    batchShape = {1};
    aBatchStrides = {0};
    bBatchStrides = {0};
  }

  if (isMatrixVector) {
    bool bIsMatrix = N != 1;
    MatrixVectorMultiply(device, dataType, batchShape, out,
                         bIsMatrix ? b : a,
                         bIsMatrix ? !bTransposed : aTransposed,
                         bIsMatrix ? (bTransposed ? bShape[bShape.size() - 1]
                                                  : bShape[bShape.size() - 2])
                                   : (aTransposed ? aShape[aShape.size() - 1]
                                                  : aShape[aShape.size() - 2]),
                         bIsMatrix ? (bTransposed ? bShape[bShape.size() - 2]
                                                  : bShape[bShape.size() - 1])
                                   : (aTransposed ? aShape[aShape.size() - 2]
                                                  : aShape[aShape.size() - 1]),
                         bIsMatrix ? bLeadingStride : aLeadingStride,
                         bIsMatrix ? bBatchStrides : aBatchStrides,
                         bIsMatrix ? a: b,
                         bIsMatrix ? aBatchStrides : bBatchStrides);
  } else {
    throw std::runtime_error("gemm kernel has not been implemented.");
  }
}

}  // namespace betann
