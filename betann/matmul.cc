#include "betann/matmul.h"

#include <fmt/format.h>

#include "betann/kernels.h"
#include "betann/kernels_helper.h"
#include "betann/preprocessor.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

bool IsTranposed(const std::vector<uint32_t>& shape,
                 const std::vector<uint32_t>& strides,
                 bool mustBeContiguous) {
  return strides[strides.size() - 2] == 1 &&
         (strides[strides.size() - 1] == shape[shape.size() - 2] ||
          !mustBeContiguous);
}

bool NeedsContiguousCopy(const std::vector<uint32_t>& shape,
                         const std::vector<uint32_t>& strides,
                         bool isMatrixVector) {
  auto stx = strides[strides.size()] - 2;
  auto sty = strides[strides.size()] - 1;
  if (!isMatrixVector)  // the gemm kernel supports non-contiguous kernel
    return stx == 1 || sty == 1;
  return (stx == 1 && sty == shape[shape.size() - 2]) ||
         (sty == 1 && stx == shape[shape.size() - 1]);
}

wgpu::Buffer CopyArray(Device& device,
                       DataType dataType,
                       const wgpu::Buffer& src,
                       const std::vector<uint32_t>& shape,
                       const std::vector<uint32_t>& strides) {
  wgpu::Buffer dst = device.CreateBuffer(
      NumElements(shape, strides) * SizeOf(dataType),
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  CopyGeneral(device, dataType, dst, dataType, src, shape, strides);
  return dst;
}

}  // namespace

void MatrixVectorMultiply(Device& device,
                          DataType dataType,
                          const std::vector<uint32_t>& batchShape,
                          const wgpu::Buffer& out,
                          const wgpu::Buffer& mat,
                          bool matTranspose,
                          uint32_t matRows,
                          uint32_t matCols,
                          const std::vector<uint32_t>& batchStridesMat,
                          const wgpu::Buffer& vec,
                          const std::vector<uint32_t>& batchStridesVec,
                          bool disableSubgroups) {
  // Figure out whether to use subgroups kernel.
  bool enableSubgroups = !disableSubgroups && device.SupportsSubgroups();
  bool enableSubgroupsF16 = false;
#ifndef __APPLE__
  if (matTranspose) {
    // There is no way to control subgroup size and it is usually too small for
    // gemvt kernel.
    enableSubgroups = false;
  }
#endif
  if (enableSubgroups && dataType == DataType::f16) {
    enableSubgroups = device.SupportsF16() && device.SupportsSubgroupsF16();
    enableSubgroupsF16 = enableSubgroups;
  }

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

  std::vector<wgpu::Buffer> args = {
      out,
      mat,
      device.CreateBufferFromScalar(matRows),
      device.CreateBufferFromScalar(matCols),
      vec,
  };
  const bool contiguous = batchShape.size() < 2;
  if (!contiguous) {
    args.push_back(device.CreateBufferFromVector(batchShape));
    args.push_back(device.CreateBufferFromVector(batchStridesMat));
    args.push_back(device.CreateBufferFromVector(batchStridesVec));
  }
  RunKernel(device,
            matTranspose ? "gemvt" : "gemv",
            fmt::format("gemv_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                        matTranspose,
                        contiguous,
                        enableSubgroups,
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
                        {"enable_f16", device.SupportsF16()},
                        {"enable_subgroups", enableSubgroups},
                        {"enable_subgroups_f16", enableSubgroupsF16},
                        {"group_count", groupCount},
                        {"group_rows", groupRows},
                        {"group_cols", groupCols},
                        {"row_work_per_thread", rowWorkPerThread},
                        {"col_work_per_thread", colWorkPerThread},
#ifdef __APPLE__
                        {"needs_workgroup_reduction", !enableSubgroups},
                        {"subgroup_min_size", 32u},
#else
                        {"needs_workgroup_reduction", true},
                        {"subgroup_min_size", 4u},
#endif
                      }),
                  wgsl_source_utils);
            },
            args,
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
                    const wgpu::Buffer& out,
                    wgpu::Buffer a,
                    const std::vector<uint32_t>& aShape,
                    const std::vector<uint32_t>& aStrides,
                    wgpu::Buffer b,
                    const std::vector<uint32_t>& bShape,
                    const std::vector<uint32_t>& bStrides) {
  if (aShape.size() < 2 || bShape.size() < 2)
    throw std::runtime_error("Inputs of MatrixMultipy must be matrices.");

  // Return 0s if either input is empty.
  if (a.GetSize() == 0 || b.GetSize() == 0) {
    CopyContiguous(device, CopyType::Scalar,
                   dataType, out, out.GetSize() / SizeOf(dataType),
                   DataType::u32, device.CreateBufferFromScalar(0u));
    return;
  }

  uint32_t M = aShape[aShape.size() - 2];
  uint32_t K = aShape[aShape.size() - 1];
  uint32_t N = bShape[bShape.size() - 1];
  bool isMatrixVector = M == 1 || N == 1;

  // Check transpose state and do contiguous copies when necessary.
  bool aTransposed, bTransposed;
  if (NeedsContiguousCopy(aShape, aStrides, isMatrixVector)) {
    aTransposed = false;
    a = CopyArray(device, dataType, a, aShape, aStrides);
  } else {
    aTransposed = IsTranposed(aShape, aStrides, isMatrixVector);
  }
  if (NeedsContiguousCopy(bShape, bStrides, isMatrixVector)) {
    bTransposed = false;
    b = CopyArray(device, dataType, b, aShape, aStrides);
  } else {
    bTransposed = IsTranposed(bShape, bStrides, isMatrixVector);
  }

  // Collapse batch dimensions.
  std::vector<uint32_t> aBatchShape(aShape.begin(), aShape.end() - 2);
  std::vector<uint32_t> bBatchShape(bShape.begin(), bShape.end() - 2);
  if (aBatchShape != bBatchShape)
    throw std::runtime_error("Matrices have incorrectly broadcasted shapes.");
  auto [batchShape, aBatchStrides, bBatchStrides] =
      CollapseContiguousDims(
          aBatchShape,
          std::vector<uint32_t>(aStrides.begin(), aStrides.end() -2),
          std::vector<uint32_t>(bStrides.begin(), bStrides.end() -2));
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
                         bIsMatrix ? bBatchStrides : aBatchStrides,
                         bIsMatrix ? a: b,
                         bIsMatrix ? aBatchStrides : bBatchStrides);
  } else {
    throw std::runtime_error("gemm kernel has not been implemented.");
  }
}

}  // namespace betann
