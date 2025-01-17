#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <string>

#include "betann/device.h"

namespace betann {

template<typename T>
inline const char* GetWgslDataType() {
  if constexpr (std::is_same_v<T, bool>)
    return "bool";
  else if constexpr (std::is_same_v<T, int32_t>)
    return "i32";
  else if constexpr (std::is_same_v<T, uint32_t>)
    return "u32";
  else if constexpr (std::is_same_v<T, float>)
    return "f32";
  else
    static_assert(false, "Unsupported C++ type in WGSL.");
}

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
