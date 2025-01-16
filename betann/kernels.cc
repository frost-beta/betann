#include "betann/kernels.h"

#include <absl/strings/substitute.h>
#include <fmt/format.h>

#include "wgsl_sources.h"

namespace betann {

void RunBinaryOp(Device& device,
                 const char* type,
                 const char* name,
                 size_t numElements,
                 const char* inputDType,
                 const wgpu::Buffer& a,
                 const wgpu::Buffer& b,
                 const char* outputDType,
                 const wgpu::Buffer& output) {
  std::string kernelName = fmt::format("binary_{}_{}", type, name);
  std::string shaderName = fmt::format("{}_{}_{}", kernelName,
                                                   inputDType,
                                                   outputDType);
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      shaderName.c_str(),
      [&]() { return GetBinaryShaderSource(name, inputDType, outputDType); });
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  float workgroupSize = 256;
  uint32_t workgroupsCount = std::ceil(numElements / workgroupSize);
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, {a, b, output}),
                   {workgroupsCount});
}

std::string GetBinaryShaderSource(const char* op,
                                  const char* inputDType,
                                  const char* outputDType) {
  constexpr std::string_view source(&wgsl_source_binary.front(),
                                    wgsl_source_binary.size());
  return absl::Substitute(source, op, inputDType, outputDType);
}

}  // namespace betann
