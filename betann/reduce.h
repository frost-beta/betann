#ifndef BETANN_REDUCE_H_
#define BETANN_REDUCE_H_

#include "betann/device.h"

namespace betann {

enum class ReduceType {
  And,
  Or,
  Sum,
  Prod,
  Min,
  Max
};

// Reduce contiguous input to one output.
void ReduceAll(Device& device,
               ReduceType type,
               DataType outputDataType,
               const Buffer& output,
               DataType inputDataType,
               const Buffer& input,
               uint32_t inputNumElements,
               bool disableSubgroups = false);

}  // namespace betann

#endif  // BETANN_REDUCE_H_
