#ifndef BETANN_REDUCE_H_
#define BETANN_REDUCE_H_

#include "betann/kernels.h"

namespace betann {

// Reduce contiguous input to one output.
void ReduceAll(Device& device,
               ReduceType type,
               DataType outputDataType,
               const Buffer& output,
               DataType inputDataType,
               const Buffer& input,
               uint32_t inputNumElements,
               bool disableSubgroups = false);

// Reduce the last dimension in contiguous input.
void ReduceLast(Device& device,
                ReduceType type,
                DataType outputDataType,
                const Buffer& output,
                uint32_t outputNumElements,
                DataType inputDataType,
                const Buffer& input,
                uint32_t rowSize,
                bool disableSubgroups = false);

// Reduce contiguous rows of input.
void ReduceRow(Device& device,
               ReduceType type,
               DataType outputDataType,
               const Buffer& output,
               uint32_t outputNumElements,
               DataType inputDataType,
               const Buffer& input,
               const std::vector<uint32_t>& inputShape,
               const std::vector<uint32_t>& inputStrides,
               const std::vector<uint32_t>& reductionAxes,
               std::vector<uint32_t> reductionShape,
               std::vector<uint32_t> reductionStrides,
               bool disableSubgroups = false);

// Write initial values to output.
void ReduceNone(Device& device,
                ReduceType type,
                DataType outputDataType,
                const Buffer& output,
                uint32_t outputNumElements);

template<typename T>
std::vector<T> RemoveIndices(const std::vector<T>& vec,
                             const std::vector<uint32_t>& indices) {
  std::vector<T> result;
  uint32_t index = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (index < indices.size() && indices[index] == i)
      index++;
    else
      result.push_back(vec[i]);
  }
  return result;
}

template<typename T>
std::vector<T> KeepIndices(const std::vector<T>& vec,
                           const std::vector<uint32_t>& axes) {
  std::vector<T> result;
  for (uint32_t axis : axes)
    result.push_back(vec[axis]);
  return result;
}

}  // namespace betann

#endif  // BETANN_REDUCE_H_
