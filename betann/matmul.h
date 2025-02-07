#ifndef BETANN_MATMUL_H_
#define BETANN_MATMUL_H_

#include "betann/device.h"
#include "betann/utils.h"

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
                          bool disableSubgroups = false);

}  // namespace betann

#endif  // BETANN_MATMUL_H_
