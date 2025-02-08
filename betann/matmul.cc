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
            fmt::format("gemv_{}_{}_{}_{}_{}",
                        WgslType(dataType),
                        contiguous,
                        workPerRow,
                        workgroupSizeRow,
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
#ifdef __APPLE__
                        {"subgroup_min_size", 32u},
                        {"needs_workgroup_reduction", !enableSubgroups},
#else
                        {"subgroup_min_size", 4u},
                        {"needs_workgroup_reduction", true},
#endif
                        {"work_per_row", workPerRow},
                        {"workgroup_size_row", workgroupSizeRow},
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
