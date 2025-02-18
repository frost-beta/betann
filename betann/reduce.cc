#include "betann/reduce.h"

#include <fmt/format.h>

#include "betann/kernels.h"
#include "betann/kernels_helper.h"
#include "betann/preprocessor.h"
#include "wgsl_sources.h"

namespace betann {

void ReduceAll(Device& device,
               const char* op,
               DataType outputDataType,
               const Buffer& output,
               DataType inputDataType,
               const Buffer& input,
               uint32_t inputNumElements) {
  const uint32_t workPerThread = 4;
  if (inputNumElements <= workPerThread * 1024) {
    RunKernel(device,
              fmt::format("reduce_all_{}", op),
              fmt::format("reduce_all_{}_{}_{}",
                          op,
                          WgslType(outputDataType),
                          WgslType(inputDataType)),
              [&]() {
                return Append(
                    ParseTemplate(wgsl_source_reduce_all,
                                  {
                                    {"enable_f16", device.SupportsF16()},
                                    {"output_dtype", WgslType(outputDataType)},
                                    {"input_dtype", WgslType(inputDataType)},
                                    {"op", op},
                                  }),
                    ParseTemplate(wgsl_source_constants,
                                  {
                                    {"enable_f16", device.SupportsF16()},
                                    {"dtype", WgslType(outputDataType)},
                                  }),
                    ParseTemplate(wgsl_source_reduce_ops,
                                  {
                                    {"output_dtype", WgslType(outputDataType)},
                                    {"op", op},
                                  }));
              },
              {output, input},
              {1, 1, 1});
  } else {
    throw std::runtime_error("ReduceAll not implemented for large input.");
  }
}

}  // namespace betann
