#ifndef SRC_DEVICE_H_
#define SRC_DEVICE_H_

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <dawn/native/EventManager.h>
#include <dawn/webgpu_cpp.h>

namespace betann {

struct Size {
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

  wgpu::Buffer CreateBuffer(wgpu::BufferUsage usage, size_t size);
  wgpu::Buffer CreateBufferFromData(wgpu::BufferUsage usage, size_t size,
                                    void* data);
  wgpu::Buffer CopyToStagingBuffer(const wgpu::Buffer& buffer);
  void CopyBufferToBuffer(const wgpu::Buffer& src, const wgpu::Buffer& dst);
  void WriteBuffer(void* data, size_t size, wgpu::Buffer* buffer);
  void ReadStagingBuffer(const wgpu::Buffer& buffer,
                         std::function<void(const void* data)> cb);

  const wgpu::ShaderModule& CreateShaderModule(const char* name,
                                               const char* source);
  const wgpu::ComputePipeline& CreateKernel(const wgpu::ShaderModule& shader,
                                            const char* entryPoint);
  wgpu::BindGroup CreateBindGroup(const wgpu::ComputePipeline& kernel,
                                  std::initializer_list<wgpu::Buffer> buffers);
  void RunKernel(const wgpu::ComputePipeline& kernel,
                 const wgpu::BindGroup& bindGroup,
                 Size gridDim);


 private:
  void EnsureEncoder();
  void EndEncoding();
  void CreateInterruptEvent();
  void WakeUpPollingThread();
  dawn::native::EventManager* GetEventManager();

  static void PollingThread(Device* self);

  // Provide an event to interrupt WaitAny calls, due to lack of API:
  // https://github.com/webgpu-native/webgpu-headers/issues/495
  struct InterruptEvent final
      : public dawn::native::EventManager::TrackedEvent {
   public:
    static dawn::Ref<InterruptEvent> Create();

    bool completed = false;

   private:
    InterruptEvent();
    ~InterruptEvent() override;

    void Complete(dawn::EventCompletionType completionType) override;
  };

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
  wgpu::Queue queue_;

  std::map<std::string, wgpu::ShaderModule> modules_;
  std::map<std::pair<WGPUShaderModule, std::string>,
           wgpu::ComputePipeline> kernels_;

  wgpu::CommandEncoder encoder_;
  std::vector<wgpu::CommandBuffer> commands_;

  std::thread polling_thread_;
  std::mutex polling_mutex_;
  std::set<uint64_t> futures_;
  bool shutdown_ = false;
  dawn::Ref<InterruptEvent> interrupt_event_;
  uint64_t interrupt_future_ = 0;
};

}  // namespace betann

#endif  // SRC_DEVICE_H_
