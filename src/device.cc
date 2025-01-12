#include "src/device.h"

#include <stdexcept>
#include <string>

#include <dawn/native/DawnNative.h>
#include <dawn/native/Instance.h>
#include <fmt/format.h>

namespace betann {

Device::Device() {
  std::array toggles = { "enable_immediate_error_handling" };
  wgpu::DawnTogglesDescriptor toggles_descriptor;
  toggles_descriptor.enabledToggles = toggles.data();
  toggles_descriptor.enabledToggleCount = toggles.size();
  wgpu::InstanceDescriptor instance_descriptor;
  instance_descriptor.features.timedWaitAnyEnable = true;
  instance_descriptor.nextInChain = &toggles_descriptor;
  instance_ = wgpu::CreateInstance(&instance_descriptor);
  if (!instance_)
    throw std::runtime_error("CreateInstance failed.");
  // Synchronously request the adapter.
  wgpu::RequestAdapterOptions options = {};
  wgpu::Future future = instance_.RequestAdapter(
      &options,
      wgpu::CallbackMode::AllowSpontaneous,
      [this](wgpu::RequestAdapterStatus status,
             wgpu::Adapter result,
             const char* message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          throw std::runtime_error(
              fmt::format("RequestAdapter failed: {0} ", message));
        }
        adapter_ = std::move(result);
      });
  instance_.WaitAny(future, 5 * 1000);
  // Synchronously request the device.
  wgpu::DeviceDescriptor device_descriptor;
  device_descriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device& device,
         wgpu::DeviceLostReason reason,
         const char* message) {
        if (reason != wgpu::DeviceLostReason::Destroyed) {
          throw std::runtime_error(
              fmt::format("Device lost: {0} ", message));
        }
      });
  device_descriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device& device,
         wgpu::ErrorType type,
         const char* message) {
        throw std::runtime_error(message);
      });
  future = adapter_.RequestDevice(
      &device_descriptor,
      wgpu::CallbackMode::AllowSpontaneous,
      [this](wgpu::RequestDeviceStatus status,
             wgpu::Device result,
             const char* message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          throw std::runtime_error(
              fmt::format("RequestDevice failed: {0} ", message));
        }
        device_ = std::move(result);
      });
  instance_.WaitAny(future, 5 * 1000);
  queue_ = device_.GetQueue();
  // The event used to notify the shutdown of process.
  shutdown_event_ = InterruptEvent::Create();
  GetEventManager()->TrackEvent(shutdown_event_);
  // The event used for interrupting ProcessPollEvents.
  CreateInterruptEvent();
  // Create a thread to poll futures.
  polling_thread_ = std::thread(&Device::PollingThread, this);
}

Device::~Device() {
  GetEventManager()->SetFutureReady(shutdown_event_.Get());
  polling_thread_.join();
}

void Device::Flush() {
  EndEncoding();
  queue_.Submit(commands_.size(), commands_.data());
  commands_.clear();
}

void Device::OnSubmittedWorkDone(std::function<void()> cb) {
  queue_.OnSubmittedWorkDone(
      wgpu::CallbackMode::AllowSpontaneous,
      [cb = std::move(cb)](wgpu::QueueWorkDoneStatus status) {
        if (status != wgpu::QueueWorkDoneStatus::Success) {
          throw std::runtime_error(
              fmt::format("OnSubmittedWorkDone failed: {0}",
                          static_cast<uint32_t>(status)));
        }
        cb();
      });
  WakeUpPollingThread();
}

wgpu::Buffer Device::CreateBuffer(wgpu::BufferUsage usage, size_t size) {
  wgpu::BufferDescriptor descriptor;
  descriptor.usage = usage;
  descriptor.size = size;
  return device_.CreateBuffer(&descriptor);
}

wgpu::Buffer Device::CreateBuffer(wgpu::BufferUsage usage, size_t size,
                                  void* data) {
  wgpu::BufferDescriptor descriptor;
  descriptor.usage = usage;
  descriptor.size = size;
  descriptor.mappedAtCreation = true;
  wgpu::Buffer buffer = device_.CreateBuffer(&descriptor);
  memcpy(buffer.GetMappedRange(), data, size);
  buffer.Unmap();
  return buffer;
}

wgpu::Buffer Device::CopyToStagingBuffer(const wgpu::Buffer& buffer) {
  wgpu::Buffer staging = CreateBuffer(wgpu::BufferUsage::MapRead |
                                      wgpu::BufferUsage::CopyDst,
                                      buffer.GetSize());
  CopyBufferToBuffer(buffer, staging);
  return staging;
}

void Device::CopyBufferToBuffer(const wgpu::Buffer& src,
                                const wgpu::Buffer& dst) {
  EnsureEncoder();
  encoder_.CopyBufferToBuffer(src, 0, dst, 0, src.GetSize());
}

void Device::WriteBuffer(void* data, size_t size, wgpu::Buffer* buffer) {
  queue_.WriteBuffer(*buffer, 0, data, size);
}

void Device::ReadStagingBuffer(const wgpu::Buffer& buffer,
                               std::function<void(const void* data)> cb) {
  buffer.MapAsync(wgpu::MapMode::Read,
                  0,
                  wgpu::kWholeMapSize,
                  wgpu::CallbackMode::AllowSpontaneous,
                  [buffer, cb=std::move(cb)](wgpu::MapAsyncStatus status,
                                             const char* message) {
                    if (status != wgpu::MapAsyncStatus::Success) {
                      throw std::runtime_error(
                          fmt::format("MapAsync failed: {0}", message));
                    }
                    cb(buffer.GetConstMappedRange());
                    buffer.Unmap();
                  });
  WakeUpPollingThread();
}

void Device::EnsureEncoder() {
  if (!encoder_)
    encoder_ = device_.CreateCommandEncoder();
}

void Device::EndEncoding() {
  if (!encoder_)
    return;
  commands_.push_back(encoder_.Finish());
  encoder_ = nullptr;
}

dawn::native::EventManager* Device::GetEventManager() {
  return dawn::native::FromAPI(instance_.Get())->GetEventManager();
}

void Device::CreateInterruptEvent() {
  DAWN_ASSERT(!interrupt_event_ || interrupt_event_->completed);
  interrupt_event_ = InterruptEvent::Create();
  GetEventManager()->TrackEvent(interrupt_event_);
}

void Device::WakeUpPollingThread() {
  std::lock_guard lock(interrup_mutext_);
  GetEventManager()->SetFutureReady(interrupt_event_.Get());
}

// static
void Device::PollingThread(Device* self) {
  while (true) {
    self->GetEventManager()->ProcessPollEvents();
    if (self->shutdown_event_->completed)
      break;
    std::lock_guard lock(self->interrup_mutext_);
    if (self->interrupt_event_->completed)
      self->CreateInterruptEvent();
  }
}

// static
dawn::Ref<Device::InterruptEvent> Device::InterruptEvent::Create() {
  return dawn::AcquireRef(new InterruptEvent());
}

Device::InterruptEvent::InterruptEvent()
    : TrackedEvent(wgpu::CallbackMode::AllowSpontaneous,
                   dawn::native::SystemEvent::CreateNonProgressingEvent()) {}

Device::InterruptEvent::~InterruptEvent() {
  EnsureComplete(dawn::EventCompletionType::Shutdown);
}

void Device::InterruptEvent::Complete(dawn::EventCompletionType) {
  completed = true;
}

}  // namespace betann
