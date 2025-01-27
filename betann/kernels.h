#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <stdexcept>
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

void BinaryOpContiguous(Device& device,
                        const char* name,
                        BinaryOpType type,
                        const char* outputDataType,
                        const wgpu::Buffer& output,
                        size_t outputNumElements,
                        const char* inputDataType,
                        const wgpu::Buffer& a,
                        const wgpu::Buffer& b);

void BinaryOpGeneral(Device& device,
                     const char* name,
                     const std::vector<uint32_t>& shape,
                     const char* outputDataType,
                     const wgpu::Buffer& output,
                     const char* inputDataType,
                     const wgpu::Buffer& a,
                     size_t aNumElements,
                     const std::vector<uint32_t>& aStrides,
                     const wgpu::Buffer& b,
                     size_t bNumElements,
                     const std::vector<uint32_t>& bStrides);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
