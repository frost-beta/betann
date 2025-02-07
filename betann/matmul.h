#ifndef BETANN_MATMUL_H_
#define BETANN_MATMUL_H_

#include "betann/device.h"
#include "betann/utils.h"

namespace betann {

// The gemv kernel with contiguous batches.
void MatrixVectorMultiply(Device& device,
                          DataType dataType,
                          const wgpu::Buffer& out,
                          const wgpu::Buffer& mat,
                          const std::vector<uint32_t>& matShape,
                          const wgpu::Buffer& vec,
                          bool disableSubgroups = false);

}  // namespace betann

#endif  // BETANN_MATMUL_H_
