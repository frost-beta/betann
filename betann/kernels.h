#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <string>

#include "betann/device.h"

namespace betann {

template<typename T>
inline const char* GetWgslDataType() {
  throw std::runtime_error("Unsupported C++ data type in WGSL.");
}
template<> inline const char* GetWgslDataType<bool>() { return "bool"; }
template<> inline const char* GetWgslDataType<int32_t>() { return "i32"; }
template<> inline const char* GetWgslDataType<uint32_t>() { return "u32"; }
template<> inline const char* GetWgslDataType<float>() { return "f32"; }

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
              const char* outputDataType,
              const wgpu::Buffer& output,
              const char* inputDataType,
              const wgpu::Buffer& a,
              const wgpu::Buffer& b);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
