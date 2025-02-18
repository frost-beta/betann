#ifndef BETANN_MATMUL_H_
#define BETANN_MATMUL_H_

#include "betann/device.h"

namespace betann {

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
                          bool disableSubgroups = false);

}  // namespace betann

#endif  // BETANN_MATMUL_H_
