#ifndef BETANN_KERNELS_HELPER_H_
#define BETANN_KERNELS_HELPER_H_

#include <string>
#include <vector>

#include "betann/device.h"
#include "betann/preprocessor.h"

namespace betann {

template<typename... Args>
inline std::string Append(std::string prefix, Args&&... args) {
  ((prefix += args), ...);
  return prefix;
}

template<typename F>
void RunKernel(Device& device,
               const std::string& kernelName,
               const std::string& shaderKey,
               F&& getSource,
               std::vector<Buffer> buffers,
               Dims3 workgroupsCount) {
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      shaderKey.c_str(),
      std::forward<F>(getSource));
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, std::move(buffers)),
                   workgroupsCount);
}

template<typename... Args>
inline bool EnableF16(Device& device, Args... dataType) {
  if (!((dataType == DataType::F16) && ...))
    return false;
  if (!device.SupportsF16())
    throw std::runtime_error("Device does not support f16 data type.");
  return true;
}

bool EnableSubgroups(Device& device, bool enableF16, bool disableSubgroups);

VariablesMap GetCapacityVariables(Device& device,
                                  bool enableF16,
                                  bool disableSubgroups);

}  // namespace betann

#endif  // BETANN_KERNELS_HELPER_H_
