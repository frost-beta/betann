#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <stdexcept>
#include <string>

#include "betann/device.h"
#include "betann/utils.h"

namespace betann {

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
                        DataType outputDataType,
                        const wgpu::Buffer& output,
                        uint32_t outputNumElements,
                        DataType inputDataType,
                        const wgpu::Buffer& a,
                        const wgpu::Buffer& b);

// Run binary operands on virtual inputs and write to contiguous output.
void BinaryOpGeneral(Device& device,
                     const char* name,
                     DataType outputDataType,
                     const wgpu::Buffer& output,
                     const std::vector<uint32_t>& shape,
                     DataType inputDataType,
                     const wgpu::Buffer& a,
                     const std::vector<uint32_t>& aStrides,
                     const wgpu::Buffer& b,
                     const std::vector<uint32_t>& bStrides);

enum class CopyType {
  // Copy a raw scalar input into the contiguous output.
  Scalar,
  // Copy the raw input buffer contiguously into a raw output buffer of the same
  // size.
  Vector,
};

// Copy data from src to dst contiguously.
void CopyContiguous(Device& device,
                    CopyType type,
                    DataType dstDataType,
                    const wgpu::Buffer& dst,
                    uint32_t dstNumElements,
                    DataType srcDataType,
                    const wgpu::Buffer& src);

// Copy data from virtual src to contiguous dst.
void CopyGeneral(Device& device,
                 DataType dstDataType,
                 const wgpu::Buffer& dst,
                 DataType srcDataType,
                 const wgpu::Buffer& src,
                 const std::vector<uint32_t>& srcShape,
                 const std::vector<uint32_t>& srcStrides);

// Copy data from virtual src to virtual dst.
void CopyGeneralBoth(Device& device,
                     DataType dstDataType,
                     const wgpu::Buffer& dst,
                     const std::vector<uint32_t>& dstStrides,
                     DataType srcDataType,
                     const wgpu::Buffer& src,
                     const std::vector<uint32_t>& srcShape,
                     const std::vector<uint32_t>& srcStrides);

// Generate random bits from contiguous keys.
void RandomBitsContiguous(Device& device,
                          DataType outDataType,
                          const wgpu::Buffer& out,
                          uint32_t outNumElements,
                          const wgpu::Buffer& keys,
                          uint32_t keysNumElements);

// Generate random bits from virual keys.
void RandomBitsGeneral(Device& device,
                       DataType outDataType,
                       const wgpu::Buffer& out,
                       uint32_t outNumElements,
                       const wgpu::Buffer& keys,
                       const std::vector<uint32_t>& keysShape,
                       const std::vector<uint32_t>& keysStrides);

enum class SortResultType {
  // Return sorted values.
  Values,
  // Return indices that would sort the values.
  Indices,
};

void SortContiguous(Device& device,
                    uint32_t axis,
                    SortResultType resultType,
                    DataType dataType,
                    const wgpu::Buffer& out,
                    const std::vector<uint32_t>& outStrides,
                    const wgpu::Buffer& input,
                    const std::vector<uint32_t>& inputShape,
                    const std::vector<uint32_t>& inputStrides);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
