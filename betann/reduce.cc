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
               uint32_t inputNumElements,
               bool disableSubgroups) {
  const uint32_t workPerThread = 4;

  // Figure out whether to use subgroups kernel.
  bool enableSubgroups = !disableSubgroups && device.SupportsSubgroups();
  bool enableSubgroupsF16 = false;
  if (enableSubgroups &&
      (outputDataType == DataType::F16 || inputDataType == DataType::F16)) {
    enableSubgroups = device.SupportsF16() && device.SupportsSubgroupsF16();
    enableSubgroupsF16 = enableSubgroups;
  }

  // Kernel creation helper.
  auto runKernel = [&](DataType outputDataType,
                       const Buffer& output,
                       DataType inputDataType,
                       const Buffer& input,
                       uint32_t workgroupSize,
                       uint32_t rowSize,
                       uint32_t numRows) {
    RunKernel(device,
              fmt::format("reduce_all_{}", op),
              fmt::format("reduce_all_{}_{}_{}_{}_{}",
                          op,
                          enableSubgroups,
                          workgroupSize,
                          WgslType(outputDataType),
                          WgslType(inputDataType)),
              [&]() {
                return Append(
                    ParseTemplate(
                        wgsl_source_reduce_all,
                        {
                          {"enable_f16", device.SupportsF16()},
                          {"enable_subgroups", enableSubgroups},
                          {"enable_subgroups_f16", enableSubgroupsF16},
                          {"op", op},
                          {"output_dtype", WgslType(outputDataType)},
                          {"input_dtype", WgslType(inputDataType)},
                          {"workgroup_size", workgroupSize},
#ifdef __APPLE__
                          {"subgroup_min_size", 32u},
#else
                          {"subgroup_min_size", 4u},
#endif
                        }),
                    ParseTemplate(
                        wgsl_source_constants,
                        {
                          {"enable_f16", device.SupportsF16()},
                          {"dtype", WgslType(outputDataType)},
                        }),
                    ParseTemplate(
                        wgsl_source_reduce_ops,
                        {
                          {"enable_subgroups", enableSubgroups},
                          {"op", op},
                          {"output_dtype", WgslType(outputDataType)},
                        }));
              },
              {output, input, device.CreateBufferFromScalar(rowSize)},
              {1, numRows, 1});
  };

  // Kernel dispatch.
  if (inputNumElements <= workPerThread * 1024) {
    // Small input use a single workgroup.
    uint32_t workgroupSize = 64;  // TODO(zcbenz): make it dynamic
    runKernel(outputDataType, output, inputDataType, input,
              workgroupSize, inputNumElements, 1);
  } else {
    // Do reduction in 2 passes.
    uint32_t numRows, workgroupSize2ndPass;
    if (inputNumElements * SizeOf(inputDataType) <= (1 << 26)) {
      numRows = workPerThread * 32;
      workgroupSize2ndPass = 32;
    } else {
      numRows = workPerThread * 1024;
      workgroupSize2ndPass = 1024;
    }
    // 1st pass.
    uint32_t rowSize = DivCeil(inputNumElements, numRows);
    uint32_t workgroupSize = 256;
    Buffer intermediate = device.CreateBuffer(numRows * SizeOf(outputDataType),
                                              BufferUsage::Storage);
    runKernel(outputDataType, intermediate, inputDataType, input,
              workgroupSize, rowSize, numRows);
    // 2nd pass.
    runKernel(outputDataType, output, outputDataType, intermediate,
              workgroupSize2ndPass, numRows, 1);
  }
}

}  // namespace betann
