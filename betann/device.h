#ifndef BETANN_DEVICE_H_
#define BETANN_DEVICE_H_

#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace betann {

struct GridDims {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

class Device {
 public:
  Device();
  ~Device();

  void Flush();
  void OnSubmittedWorkDone(std::function<void()> cb);

  wgpu::Buffer CreateBuffer(size_t size, wgpu::BufferUsage usage);
  wgpu::Buffer CreateBufferFromData(
      const void* data,
      size_t size,
      wgpu::BufferUsage usage = wgpu::BufferUsage::Storage);
  template<typename T>
  wgpu::Buffer CreateBufferFromVector(
      const std::vector<T>& vec,
      wgpu::BufferUsage usage = wgpu::BufferUsage::Storage) {
    return CreateBufferFromData(vec.data(), vec.size() * sizeof(T), usage);
  }

  wgpu::Buffer CopyToStagingBuffer(const wgpu::Buffer& buffer);
  void CopyBufferToBuffer(const wgpu::Buffer& src, const wgpu::Buffer& dst);
  void WriteBuffer(void* data, size_t size, wgpu::Buffer* buffer);
  void ReadStagingBuffer(const wgpu::Buffer& buffer,
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
                 GridDims gridDims);

  const wgpu::Limits& GetLimits() const { return limits_; }

 private:
  void EnsureEncoder();
  void EndEncoding();
  void AddFuture(const wgpu::Future& future);

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

  // The polling thread that wait and process events.
  std::thread thread_;
  bool shutdown_ = false;
  // Bookkeeping the futures returned by wgpu APIs.
  std::mutex mutex_;
  std::set<uint64_t> futures_;
  // Hold the polling thread when there is no futures. Can be removed after this
  // feature gets implemented in dawn:
  // https://github.com/webgpu-native/webgpu-headers/issues/495
  std::condition_variable hold_;
};

}  // namespace betann

#endif  // BETANN_DEVICE_H_
