#ifndef BETANN_REDUCE_H_
#define BETANN_REDUCE_H_

#include "betann/device.h"

namespace betann {

// Reduce contiguous input to one output.
void ReduceAll(Device& device,
               const char* op,
               DataType outputDataType,
               const Buffer& output,
               DataType inputDataType,
               const Buffer& input,
               uint32_t inputNumElements);

}  // namespace betann

#endif  // BETANN_REDUCE_H_
