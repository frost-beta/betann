#include "betann/kernels.h"

#include <absl/strings/substitute.h>
#include <fmt/format.h>

#include "wgsl_sources.h"

namespace betann {

namespace {

inline const char* GetBinaryOpTypeStr(BinaryOpType type, bool largeArray) {
  switch (type) {
    case BinaryOpType::ScalarScalar:
      return "ss";
    case BinaryOpType::ScalarVector:
      return largeArray ? "sv2" : "sv";
    case BinaryOpType::VectorScalar:
      return largeArray ? "vs2" : "vs";
    case BinaryOpType::VectorVector:
      return largeArray ? "vv2" : "vv";
  }
}

std::string GetShaderSourceBinary(const char* op,
                                  const char* inputDataType,
                                  const char* outputDataType) {
  std::string preprocessed = absl::Substitute(wgsl_source_binary,
                                              op,
                                              inputDataType,
                                              outputDataType);
  preprocessed += wgsl_source_binary_ops;
  return preprocessed;
}

}  // namespace

void BinaryOp(Device& device,
              BinaryOpType type,
              const char* name,
              size_t outputSize,
              const char* outputDataType,
              const wgpu::Buffer& output,
              const char* inputDataType,
              const wgpu::Buffer& a,
              const wgpu::Buffer& b) {
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension;
  bool use2DGrid = outputSize > maxThreadsPerGridDim;
  std::string kernelName = fmt::format("binary_{}_{}",
                                       GetBinaryOpTypeStr(type, use2DGrid),
                                       name);
  std::string shaderName = fmt::format("{}_{}_{}",
                                       kernelName,
                                       inputDataType,
                                       outputDataType);
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      shaderName.c_str(),
      [&]() {
        return GetShaderSourceBinary(name, inputDataType, outputDataType);
      });
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  if (use2DGrid) {
    gridDims.x =
        std::floor(maxThreadsPerGridDim / static_cast<float>(workgroupSize));
    gridDims.y = std::ceil(outputSize / static_cast<float>(gridDims.x));
  } else {
    gridDims.x = std::ceil(outputSize / static_cast<float>(workgroupSize));
  }
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, {a, b, output}),
                   gridDims);
}

}  // namespace betann
