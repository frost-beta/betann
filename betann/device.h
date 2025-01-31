#ifndef BETANN_DEVICE_H_
#define BETANN_DEVICE_H_

#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace betann {

struct Dims3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

class Device {
 public:
  Device();
  ~Device();

  wgpu::Future OnSubmittedWorkDone(std::function<void()> cb);
  void Flush();
  void WaitFor(const wgpu::Future& future);
  void WaitAll();

  wgpu::Buffer CreateBuffer(size_t size, wgpu::BufferUsage usage);
  wgpu::Buffer CreateBufferFromData(
      const void* data,
      size_t size,
      wgpu::BufferUsage usage = wgpu::BufferUsage::Storage);
  template<typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
  wgpu::Buffer CreateBufferFromScalar(
      T data,
      wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform) {
    return CreateBufferFromData(&data, sizeof(T), usage);
  }
  template<typename T>
  wgpu::Buffer CreateBufferFromVector(
      const std::vector<T>& vec,
      wgpu::BufferUsage usage = wgpu::BufferUsage::Storage) {
    return CreateBufferFromData(vec.data(), vec.size() * sizeof(T), usage);
  }

  wgpu::Buffer CopyToStagingBuffer(const wgpu::Buffer& buffer);
  void CopyBufferToBuffer(const wgpu::Buffer& src, const wgpu::Buffer& dst);
  void WriteBuffer(void* data, size_t size, wgpu::Buffer* buffer);
  wgpu::Future ReadStagingBuffer(const wgpu::Buffer& buffer,
                                 std::function<void(const void* data)> cb);

  const wgpu::ShaderModule& CreateShaderModule(
      const char* name,
      std::function<std::string()> getSource);
  const wgpu::ComputePipeline& CreateKernel(const wgpu::ShaderModule& shader,
                                            const char* entryPoint);
  wgpu::BindGroup CreateBindGroup(const wgpu::ComputePipeline& kernel,
                                  std::initializer_list<wgpu::Buffer> buffers);
  void RunKernel(const wgpu::ComputePipeline& kernel,
                 const wgpu::BindGroup& bindGroup,
                 Dims3 workgroupsCount);

  const wgpu::Limits& GetLimits() const { return limits_; }

 private:
  void EnsureEncoder();
  void EndEncoding();
  wgpu::Future AddFuture(const wgpu::Future& future);

  static void PollingThread(Device* self);

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
  wgpu::Queue queue_;
  wgpu::Limits limits_;

  std::map<std::string, wgpu::ShaderModule> modules_;
  std::map<std::pair<WGPUShaderModule, std::string>,
           wgpu::ComputePipeline> kernels_;

  wgpu::CommandEncoder encoder_;
  std::vector<wgpu::CommandBuffer> commands_;

  std::set<uint64_t> futures_;
};

// Allow users to avoid the wgpu namespace in their code.
using Buffer = wgpu::Buffer;
using BufferUsage = wgpu::BufferUsage;

}  // namespace betann

#endif  // BETANN_DEVICE_H_
