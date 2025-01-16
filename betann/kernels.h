#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <string>

#include "betann/device.h"

namespace betann {

enum class BinaryOpType {
  ScalarScalar,
  ScalarVector,
  VectorScalar,
  VectorVector,
};

void BinaryOp(Device& device,
              BinaryOpType type,
              const char* name,
              size_t outputSize,
              const char* outputDType,
              const wgpu::Buffer& output,
              const char* inputDType,
              const wgpu::Buffer& a,
              const wgpu::Buffer& b);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
