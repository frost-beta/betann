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
                          uint32_t matRows,
                          uint32_t matCols,
                          const std::vector<uint32_t>& batchStridesMat,
                          const wgpu::Buffer& vec,
                          const std::vector<uint32_t>& batchStridesVec,
                          bool disableSubgroups) {
  bool enableSubgroups = !disableSubgroups && device.SupportsSubgroups();
  bool enableSubgroupsF16 = false;
  if (enableSubgroups && dataType == DataType::f16) {
    enableSubgroups = device.SupportsF16() && device.SupportsSubgroupsF16();
    enableSubgroupsF16 = enableSubgroups;
  }
  const uint32_t workPerRow = matRows < 4 ? 1 : 4;
  const uint32_t workgroupSizeRow = matRows >= 4096 ? 8 : 4;
  // The subgroups gemv kernel assumes num_threads per row equals to
  // subgroups_size, so we don't need to have a shared memory for accumulations.
  // For metal we can assume subgroups_size is 32, but for other platforms
  // we have to pick the minimum value that subgroups_size can be.
#ifdef __APPLE__
  const uint32_t workgroupSizeCol = 32;
#else
  const uint32_t workgroupSizeCol = enableSubgroups ? 4 : 32;
#endif
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
            "gemv",
            fmt::format("gemv_{}_{}_{}_{}_{}_{}",
                        WgslType(dataType),
                        contiguous,
                        workPerRow,
                        workgroupSizeRow,
                        workgroupSizeCol,
                        enableSubgroups),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_gemv,
                      {
                        {"contiguous", contiguous},
                        {"dtype", WgslType(dataType)},
                        {"dtype_is_floating", IsFloating(dataType)},
                        {"enable_f16", device.SupportsF16()},
                        {"enable_subgroups", enableSubgroups},
                        {"enable_subgroups_f16", enableSubgroupsF16},
                        {"work_per_row", workPerRow},
                        {"workgroup_size_row", workgroupSizeRow},
                        {"workgroup_size_col", workgroupSizeCol},
                      }),
                  wgsl_source_utils);
            },
            args,
            {
              DivCeil(matRows, workPerRow * workgroupSizeRow),
              1,
              NumElements(batchShape),
            });
}

}  // namespace betann
