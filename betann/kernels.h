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
  // Both operands are scalars.
  ScalarScalar,
  // Left operand is a scalar and right operand is contiguous.
  ScalarVector,
  // Left operand is contiguous and right operand is a scalar.
  VectorScalar,
  // Both operands are contiguous.
  VectorVector,
};

// Run binary operations on contiguous inputs.
void BinaryOpContiguous(Device& device,
                        const char* name,
                        BinaryOpType type,
                        const char* outputDataType,
                        const wgpu::Buffer& output,
                        size_t outputNumElements,
                        const char* inputDataType,
                        const wgpu::Buffer& a,
                        const wgpu::Buffer& b);

// Run binary operands on virtual inputs and write to full contiguous output.
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

enum class CopyType {
  // Copy a raw scalar input into the full contiguous output.
  Scalar,
  // Copy the raw input buffer contiguously into a raw output buffer of the same
  // size.
  Vector,
};

// Copy data from src to dst contiguously.
void CopyContiguous(Device& device,
                    CopyType type,
                    const char* dstDataType,
                    const wgpu::Buffer& dst,
                    size_t dstNumElements,
                    const char* srcDataType,
                    const wgpu::Buffer& src);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
