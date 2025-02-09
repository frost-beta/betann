#include "betann/matmul.h"

#include <fmt/format.h>

#include "betann/preprocessor.h"
#include "betann/kernels_helper.h"
#include "wgsl_sources.h"

namespace betann {

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

}  // namespace betann
