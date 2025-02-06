#ifndef BETANN_KERNELS_HELPER_H_
#define BETANN_KERNELS_HELPER_H_

#include <string>
#include <vector>

#include "betann/device.h"

namespace betann {

template<typename F>
void RunKernel(Device& device,
               const std::string& kernelName,
               const std::string& shaderKey,
               F&& getSource,
               std::vector<wgpu::Buffer> buffers,
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

}  // namespace betann

#endif  // BETANN_KERNELS_HELPER_H_
