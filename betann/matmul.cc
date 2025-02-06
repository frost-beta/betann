#include "betann/matmul.h"

#include <fmt/format.h>

#include "betann/preprocessor.h"
#include "betann/kernels_helper.h"
#include "wgsl_sources.h"

namespace betann {

void MatrixVectorMultiply(Device& device,
                          DataType dataType,
                          const wgpu::Buffer& out,
                          const wgpu::Buffer& mat,
                          const std::vector<uint32_t>& matShape,
                          const wgpu::Buffer& vec) {
  if (matShape.size() < 2)
    throw std::runtime_error("Invalid matrix shape.");
  if (matShape.size() > 3)
    throw std::runtime_error("Non-contigous matrix is not supported in gemv.");
  uint32_t matRows = matShape[matShape.size() - 2];
  uint32_t matCols = matShape[matShape.size() - 1];
  uint32_t batches = NumElements(matShape) / (matRows * matCols);
  const uint32_t workPerRow = matRows < 4 ? 1 : 4;
  const uint32_t workgroupSizeRow = matRows >= 4096 ? 8 : 4;
  RunKernel(device,
            "gemv",
            fmt::format("gemv_{}_{}_{}",
                        WgslType(dataType), workPerRow, workgroupSizeRow),
            [&]() {
              return ParseTemplate(wgsl_source_gemv,
                                   {
                                     {"enable_f16", device.SupportsF16()},
                                     {"dtype", WgslType(dataType)},
                                     {"work_per_row", workPerRow},
                                     {"workgroup_size_row", workgroupSizeRow},
                                   });
            },
            {
              out,
              mat,
              device.CreateBufferFromScalar(matRows),
              device.CreateBufferFromScalar(matCols),
              vec,
            },
            {DivCeil(matRows, workPerRow * workgroupSizeRow), 1, batches});
}

}  // namespace betann
