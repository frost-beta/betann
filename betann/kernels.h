#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include "betann/device.h"
#include "betann/utils.h"

namespace betann {

// Generate evenly spaced values.
void ArrayRange(Device& device,
                double start,
                double step,
                DataType dataType,
                const Buffer& out);

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
                        const Buffer& output,
                        uint32_t outputNumElements,
                        DataType inputDataType,
                        const Buffer& a,
                        const Buffer& b);

// Run binary operands on virtual inputs and write to contiguous output.
void BinaryOpGeneral(Device& device,
                     const char* name,
                     DataType outputDataType,
                     const Buffer& output,
                     const std::vector<uint32_t>& shape,
                     DataType inputDataType,
                     const Buffer& a,
                     const std::vector<uint32_t>& aStrides,
                     const Buffer& b,
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
                    const Buffer& dst,
                    uint32_t dstNumElements,
                    DataType srcDataType,
                    const Buffer& src);

// Copy data from virtual src to contiguous dst.
void CopyGeneral(Device& device,
                 DataType dstDataType,
                 const Buffer& dst,
                 DataType srcDataType,
                 const Buffer& src,
                 const std::vector<uint32_t>& srcShape,
                 const std::vector<uint32_t>& srcStrides);

// Copy data from virtual src to virtual dst.
void CopyGeneralBoth(Device& device,
                     DataType dstDataType,
                     const Buffer& dst,
                     const std::vector<uint32_t>& dstStrides,
                     DataType srcDataType,
                     const Buffer& src,
                     const std::vector<uint32_t>& srcShape,
                     const std::vector<uint32_t>& srcStrides);

// Multiply contiguous matrix.
void MatrixMultiply(Device& device,
                    DataType dataType,
                    const Buffer& out,
                    Buffer a,
                    const std::vector<uint32_t>& aShape,
                    const std::vector<uint32_t>& aStrides,
                    Buffer b,
                    const std::vector<uint32_t>& bShape,
                    const std::vector<uint32_t>& bStrides);

// Generate random bits from contiguous keys.
void RandomBitsContiguous(Device& device,
                          DataType outDataType,
                          const Buffer& out,
                          uint32_t outNumElements,
                          const Buffer& keys,
                          uint32_t keysNumElements);

// Generate random bits from virual keys.
void RandomBitsGeneral(Device& device,
                       DataType outDataType,
                       const Buffer& out,
                       uint32_t outNumElements,
                       const Buffer& keys,
                       const std::vector<uint32_t>& keysShape,
                       const std::vector<uint32_t>& keysStrides);

enum class ReductionPlanType {
  ReduceAll,
  ReduceRow,
  ReduceCol,
  ReduceGeneral,
};

struct ReductionPlan {
  ReductionPlanType type;
  std::vector<uint32_t> reductionShape;
  std::vector<uint32_t> reductionStrides;
};

enum class ReduceType {
  And,
  Or,
  Sum,
  Prod,
  Min,
  Max
};

void Reduce(Device& device,
            ReductionPlan plan,
            ReduceType type,
            DataType outputDataType,
            const Buffer& output,
            uint32_t outputNumElements,
            DataType inputDataType,
            const Buffer& input,
            uint32_t inputNumElements,
            const std::vector<uint32_t>& inputShape,
            const std::vector<uint32_t>& inputStrides,
            const std::vector<uint32_t>& reductionAxes);

// Maximum number of elements can sort in one block.
uint32_t SortBlockSize();

enum class SortInputType {
  // The input is contiguous and the sorted axis has either largest or smallest
  // stride.
  Contiguous,
  // The input is virtual.
  General,
};

enum class SortResultType {
  // Return sorted values.
  Values,
  // Return indices that would sort the values.
  Indices,
};

// Sort input assuming elements in axis can fit in one block.
void SortBlock(Device& device,
               uint32_t axis,
               SortInputType inputType,
               SortResultType resultType,
               const Buffer& out,
               const std::vector<uint32_t>& outStrides,
               DataType inputDataType,
               const Buffer& input,
               const std::vector<uint32_t>& inputShape,
               const std::vector<uint32_t>& inputStrides);

// Run unary operations on contiguous input.
void UnaryOpContiguous(Device& device,
                       const char* name,
                       DataType outputDataType,
                       const Buffer& output,
                       DataType inputDataType,
                       const Buffer& input,
                       uint32_t inputNumElements);

// Run unary operations on virual input.
void UnaryOpGeneral(Device& device,
                    const char* name,
                    DataType outputDataType,
                    const Buffer& output,
                    DataType inputDataType,
                    const Buffer& input,
                    const std::vector<uint32_t>& inputShape,
                    const std::vector<uint32_t>& inputStrides);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
