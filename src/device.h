#ifndef SRC_DEVICE_H_
#define SRC_DEVICE_H_

#include <mutex>
#include <thread>

#include <dawn/native/EventManager.h>
#include <dawn/webgpu_cpp.h>

namespace betann {

class Device {
 public:
  Device();
  ~Device();

  void Flush();
  void OnSubmittedWorkDone(std::function<void()> cb);

  wgpu::Buffer CreateBuffer(wgpu::BufferUsage usage, size_t size);
  wgpu::Buffer CreateBuffer(wgpu::BufferUsage usage, size_t size, void* data);
  wgpu::Buffer CopyToStagingBuffer(const wgpu::Buffer& buffer);
  void CopyBufferToBuffer(const wgpu::Buffer& src, const wgpu::Buffer& dst);
  void WriteBuffer(void* data, size_t size, wgpu::Buffer* buffer);
  void ReadStagingBuffer(const wgpu::Buffer& buffer,
                         std::function<void(const void* data)> cb);

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

  wgpu::CommandEncoder encoder_;
  std::vector<wgpu::CommandBuffer> commands_;

  std::thread polling_thread_;
  dawn::Ref<InterruptEvent> shutdown_event_;
  std::mutex interrup_mutext_;
  dawn::Ref<InterruptEvent> interrupt_event_;
};

}  // namespace betann

#endif  // SRC_DEVICE_H_
