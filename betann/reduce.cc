#include "betann/reduce.h"

#include <fmt/format.h>

#include "betann/kernels_helper.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

const char* ReduceTypeToString(ReduceType type) {
  switch (type) {
    case ReduceType::And:
      return "and";
    case ReduceType::Or:
      return "or";
    case ReduceType::Sum:
      return "sum";
    case ReduceType::Prod:
      return "prod";
    case ReduceType::Min:
      return "min";
    case ReduceType::Max:
      return "max";
  }
}

std::string GetReduceShaderCode(const char* source,
                                const char* op,
                                const VariablesMap& capacities,
                                DataType outputDataType,
                                DataType inputDataType,
                                uint32_t workgroupSize) {
  return Append(
      ParseTemplate(
          source,
          {
            {"op", op},
            {"output_dtype", WgslType(outputDataType)},
            {"input_dtype", WgslType(inputDataType)},
            {"workgroup_size", workgroupSize},
          },
          capacities),
      ParseTemplate(
          wgsl_source_constants,
          {
            {"dtype", WgslType(outputDataType)},
          },
          capacities),
      ParseTemplate(
          wgsl_source_reduce_ops,
          {
            {"op", op},
            {"output_dtype", WgslType(outputDataType)},
          },
          capacities));
}

}  // namespace

void ReduceAll(Device& device,
               ReduceType type,
               DataType outputDataType,
               const Buffer& output,
               DataType inputDataType,
               const Buffer& input,
               uint32_t inputNumElements,
               bool disableSubgroups) {
  // Kernel creation helper.
  auto runKernel = [&](DataType outputDataType,
                       const Buffer& output,
                       DataType inputDataType,
                       const Buffer& input,
                       uint32_t workgroupSize,
                       uint32_t rowSize,
                       uint32_t numRows) {
    const char* op = ReduceTypeToString(type);
    bool enableF16 = EnableF16(device, outputDataType, inputDataType);
    auto capacities = GetCapacityVariables(device, enableF16, disableSubgroups);
    RunKernel(device,
              fmt::format("reduce_all_{}", op),
              fmt::format("reduce_all_{}_{}_{}_{}_{}",
                          op,
                          std::get<bool>(capacities["enable_subgroups"]),
                          workgroupSize,
                          WgslType(outputDataType),
                          WgslType(inputDataType)),
              [&]() {
                return GetReduceShaderCode(wgsl_source_reduce_all,
                                           op,
                                           capacities,
                                           outputDataType,
                                           inputDataType,
                                           workgroupSize);
              },
              {output, input, device.CreateBufferFromScalar(rowSize)},
              {1, numRows, 1});
  };

  // Kernel dispatch.
  const uint32_t workPerThread = 4;
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

void ReduceLast(Device& device,
                ReduceType type,
                DataType outputDataType,
                const Buffer& output,
                uint32_t outputNumElements,
                DataType inputDataType,
                const Buffer& input,
                uint32_t rowSize,
                bool disableSubgroups) {
  const char* op = ReduceTypeToString(type);
  bool enableF16 = EnableF16(device, outputDataType, inputDataType);
  auto capacities = GetCapacityVariables(device, enableF16, disableSubgroups);

  const uint32_t writePerThread = 4;
  uint32_t workgroupSize;
  if (rowSize <= 512)
    workgroupSize = 32;
  else if (rowSize <= 1024)
    workgroupSize = 128;
  else
    workgroupSize = 256;
  RunKernel(device,
            fmt::format("reduce_last_{}", op),
            fmt::format("reduce_last_{}_{}_{}_{}_{}",
                        op,
                        std::get<bool>(capacities["enable_subgroups"]),
                        workgroupSize,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return GetReduceShaderCode(wgsl_source_reduce_last,
                                         op,
                                         capacities,
                                         outputDataType,
                                         inputDataType,
                                         workgroupSize);
            },
            {
              output,
              device.CreateBufferFromScalar(outputNumElements),
              input,
              device.CreateBufferFromScalar(rowSize),
            },
            {1, DivCeil(outputNumElements, writePerThread), 1});
}

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
               bool disableSubgroups) {
  if (reductionStrides.back() != 1)
    throw std::runtime_error("The reducted row must be contiguous.");
  // The info used for reading rows.
  uint32_t rowSize = reductionShape.back();
  reductionShape.pop_back();
  reductionStrides.pop_back();
  uint32_t nonRowReductions = NumElements(reductionShape);
  // The shape/strides used for locating where to read rows.
  auto nonReductionShape = RemoveIndices(inputShape, reductionAxes);
  auto nonReductionStrides = RemoveIndices(inputStrides, reductionAxes);
  std::tie(nonReductionShape, nonReductionStrides) =
      CollapseContiguousDims(nonReductionShape, nonReductionStrides);

  // Kernel options.
  const char* op = ReduceTypeToString(type);
  bool enableF16 = EnableF16(device, outputDataType, inputDataType);
  auto capacities = GetCapacityVariables(device, enableF16, disableSubgroups);
  uint32_t coordCacheSize;
  if (reductionShape.size() <= 1)
    coordCacheSize = 1;
  else if (reductionShape.size() == 2)
    coordCacheSize = 2;
  else
    coordCacheSize = 5;
  capacities["coord_cache_size"] = coordCacheSize;
  // FIXME(zcbenz): Enable for all after upstream fixes:
  // https://issues.chromium.org/issues/398275914
  bool useFastIndex =
      device.GetAdapterInfo().backendType != wgpu::BackendType::D3D11 &&
      device.GetAdapterInfo().backendType != wgpu::BackendType::D3D12;
  capacities["use_fast_index"] = useFastIndex;

  // Kernel dispatch.
  uint32_t workgroupSize = 128;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            fmt::format("reduce_row_small_{}", op),
            fmt::format("reduce_row_small_{}_{}_{}_{}_{}_{}",
                        op,
                        std::get<bool>(capacities["enable_subgroups"]),
                        workgroupSize,
                        coordCacheSize,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return Append(GetReduceShaderCode(wgsl_source_reduce_row,
                                                op,
                                                capacities,
                                                outputDataType,
                                                inputDataType,
                                                workgroupSize),
                            wgsl_source_utils);
            },
            {
              output,
              device.CreateBufferFromScalar(outputNumElements),
              input,
              device.CreateBufferFromScalar(rowSize),
              device.CreateBufferFromScalar(nonRowReductions),
              nonReductionShape.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(nonReductionShape),
              nonReductionStrides.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(nonReductionStrides),
              reductionShape.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(reductionShape),
              reductionStrides.empty()
                  ? device.CreateBufferFromScalar(0u)
                  : device.CreateBufferFromVector(reductionStrides),
            },
            {
              DivCeil(outputNumElements, workgroupSize),
            });
}

}  // namespace betann
