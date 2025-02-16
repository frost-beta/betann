#ifndef BETANN_DEVICE_H_
#define BETANN_DEVICE_H_

#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "betann/buffer.h"
#include "betann/data_type.h"
#include "betann/utils.h"

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

  Buffer CreateBuffer(uint64_t size, BufferUsage usage);
  Buffer CreateBufferFromData(const void* data,
                              uint64_t size,
                              BufferUsage usage = BufferUsage::Storage);

  template<typename T>
  Buffer CreateBufferFromStruct(const T& obj,
                                BufferUsage usage = BufferUsage::Storage) {
    return CreateBufferFromData(&obj, sizeof(T), usage);
  }

  template<typename T>
  Buffer CreateBufferFromVector(const std::vector<T>& vec,
                                DataType dataType = GetDataType<T>(),
                                BufferUsage usage = BufferUsage::Storage) {
    assert(sizeof(T) == SizeOf(dataType));
    return CreateBufferFromData(vec.data(), vec.size() * sizeof(T), usage);
  }

  template<typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
  Buffer CreateBufferFromScalar(T data,
                                DataType dataType = GetDataType<T>(),
                                BufferUsage usage = BufferUsage::Uniform) {
    if (sizeof(T) == SizeOf(dataType))
      return CreateBufferFromData(&data, sizeof(T), usage);
    switch (dataType) {
      case DataType::Bool:
      case DataType::U32: {
        uint32_t native = static_cast<uint32_t>(data);
        return CreateBufferFromData(&native, SizeOf(dataType), usage);
      }
      case DataType::I32: {
        int32_t native = static_cast<int32_t>(data);
        return CreateBufferFromData(&native, SizeOf(dataType), usage);
      }
      case DataType::F32: {
        float native = static_cast<float>(data);
        return CreateBufferFromData(&native, SizeOf(dataType), usage);
      }
      case DataType::F16: {
        uint64_t native = Float32ToFloat16(static_cast<float>(data));
        return CreateBufferFromData(&native, SizeOf(dataType), usage);
      }
    }
  }

  void WriteBuffer(void* data, uint64_t size, Buffer& buffer);

  // Read the buffer by copying to a staging buffer first and then mapping,
  // simulnateous reads will be merged into one. This method is NOT thread-safe.
  using ReadBufferCallback = std::function<void(const void* data,
                                                uint64_t size,
                                                uint64_t offset)>;
  wgpu::Future ReadBuffer(const Buffer& buffer, ReadBufferCallback cb);

  const wgpu::ShaderModule& CreateShaderModule(
      const char* name,
      std::function<std::string()> getSource);
  const wgpu::ComputePipeline& CreateKernel(const wgpu::ShaderModule& shader,
                                            const char* entryPoint);
  wgpu::BindGroup CreateBindGroup(const wgpu::ComputePipeline& kernel,
                                  std::vector<Buffer> buffers);
  void RunKernel(const wgpu::ComputePipeline& kernel,
                 const wgpu::BindGroup& bindGroup,
                 Dims3 workgroupsCount);

  const wgpu::Limits& GetLimits() const { return limits_; }
  bool SupportsF16() const { return supportsF16_; }
  bool SupportsSubgroups() const { return supportsSubgroups_; }
  bool SupportsSubgroupsF16() const { return supportsSubgroupsF16_; }

 private:
  void EnsureEncoder();
  void EndEncoding();
  Buffer CopyToStagingBuffer(const Buffer& buffer);
  wgpu::Future AddFuture(const wgpu::Future& future);

  static void PollingThread(Device* self);

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
  wgpu::Queue queue_;

  // Device capacity.
  wgpu::Limits limits_;
  bool supportsF16_ = false;
  bool supportsSubgroups_ = false;
  bool supportsSubgroupsF16_ = false;

  // Cached shaders and kernels.
  std::map<std::string, wgpu::ShaderModule> modules_;
  std::map<std::pair<WGPUShaderModule, std::string>,
           wgpu::ComputePipeline> kernels_;

  // Current command encoder and unsubmitted commands.
  wgpu::CommandEncoder encoder_;
  std::vector<wgpu::CommandBuffer> commands_;

  // Buffers being read.
  std::map<WGPUBuffer,
           std::pair<wgpu::Future,
                     std::vector<ReadBufferCallback>>> pendingReadBuffers_;

  // Tasks to be waited for.
  std::set<uint64_t> futures_;
};

}  // namespace betann

#endif  // BETANN_DEVICE_H_
