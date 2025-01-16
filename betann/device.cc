#include "betann/device.h"

#include <array>
#include <stdexcept>

#include <fmt/format.h>

namespace betann {

Device::Device() {
  // Intialize instance with feature flags.
  std::array toggles = {
    // Throw error immediately instead of delaying to next tick.
    "enable_immediate_error_handling",
    // Enable f16 dtype.
    "shader-f16",
  };
  wgpu::DawnTogglesDescriptor togglesDescriptor;
  togglesDescriptor.enabledToggles = toggles.data();
  togglesDescriptor.enabledToggleCount = toggles.size();
  wgpu::InstanceDescriptor instanceDescriptor;
  instanceDescriptor.features.timedWaitAnyEnable = true;
  instanceDescriptor.nextInChain = &togglesDescriptor;
  instance_ = wgpu::CreateInstance(&instanceDescriptor);
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
              fmt::format("RequestAdapter failed: {}", message));
        }
        adapter_ = std::move(result);
      });
  instance_.WaitAny(future, 5 * 1000);
  // Check if there is a valid backend.
  wgpu::AdapterInfo info;
  if (adapter_.GetInfo(&info) != wgpu::Status::Success)
    throw std::runtime_error("GetInfo failed.");
  if (info.backendType == wgpu::BackendType::Null)
    throw std::runtime_error("There is no valid backend.");

  // Synchronously request the device.
  wgpu::DeviceDescriptor deviceDescriptor;
  deviceDescriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device& device,
         wgpu::DeviceLostReason reason,
         const char* message) {
        if (reason != wgpu::DeviceLostReason::Destroyed)
          throw std::runtime_error(fmt::format("Device lost: {}", message));
      });
  deviceDescriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device& device,
         wgpu::ErrorType type,
         const char* message) {
        throw std::runtime_error(message);
      });
  future = adapter_.RequestDevice(
      &deviceDescriptor,
      wgpu::CallbackMode::AllowSpontaneous,
      [this](wgpu::RequestDeviceStatus status,
             wgpu::Device result,
             const char* message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          throw std::runtime_error(
              fmt::format("RequestDevice failed: {}", message));
        }
        device_ = std::move(result);
      });
  instance_.WaitAny(future, 5 * 1000);
  queue_ = device_.GetQueue();

  // Get limitations.
  wgpu::SupportedLimits limits;
  if (device_.GetLimits(&limits) != wgpu::Status::Success)
    throw std::runtime_error("GetLimits failed.");
  limits_ = limits.limits;

  // Create a thread to poll futures.
  thread_ = std::thread(&Device::PollingThread, this);
}

Device::~Device() {
  {
    std::lock_guard lock(mutex_);
    shutdown_ = true;
  }
  hold_.notify_one();
  thread_.join();
}

void Device::Flush() {
  EndEncoding();
  if (commands_.empty()) {
    instance_.ProcessEvents();
  } else {
    queue_.Submit(commands_.size(), commands_.data());
    commands_.clear();
  }
}

void Device::OnSubmittedWorkDone(std::function<void()> cb) {
  AddFuture(queue_.OnSubmittedWorkDone(
      wgpu::CallbackMode::AllowSpontaneous,
      [cb = std::move(cb)](wgpu::QueueWorkDoneStatus status) {
        if (status != wgpu::QueueWorkDoneStatus::Success) {
          throw std::runtime_error(
              fmt::format("OnSubmittedWorkDone failed: {}",
                          static_cast<uint32_t>(status)));
        }
        cb();
      }));
}

wgpu::Buffer Device::CreateBuffer(wgpu::BufferUsage usage, size_t size) {
  wgpu::BufferDescriptor descriptor;
  descriptor.usage = usage;
  descriptor.size = size;
  return device_.CreateBuffer(&descriptor);
}

wgpu::Buffer Device::CreateBufferFromData(wgpu::BufferUsage usage, size_t size,
                                          const void* data) {
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
  AddFuture(buffer.MapAsync(
      wgpu::MapMode::Read,
      0,
      wgpu::kWholeMapSize,
      wgpu::CallbackMode::AllowSpontaneous,
      [buffer, cb=std::move(cb)](wgpu::MapAsyncStatus status,
                                 const char* message) {
        if (status != wgpu::MapAsyncStatus::Success)
          throw std::runtime_error(fmt::format("MapAsync failed: {}", message));
        cb(buffer.GetConstMappedRange());
        buffer.Unmap();
      }));
}

const wgpu::ShaderModule& Device::CreateShaderModule(
    const char* name,
    std::function<std::string()> getSource) {
  auto it = modules_.find(name);
  if (it != modules_.end())
    return it->second;
  std::string source = getSource();
  wgpu::ShaderSourceWGSL wgsl;
  wgsl.code = source.c_str();
  wgpu::ShaderModuleDescriptor descriptor;
  descriptor.label = name;
  descriptor.nextInChain = &wgsl;
  return modules_[name] = device_.CreateShaderModule(&descriptor);
}

const wgpu::ComputePipeline& Device::CreateKernel(
    const wgpu::ShaderModule& shader,
    const char* entryPoint) {
  if (!entryPoint)
    throw std::runtime_error("entryPoint must be passed in CreateKernel.");
  auto key = std::make_pair(shader.Get(), std::string(entryPoint));
  auto it = kernels_.find(key);
  if (it != kernels_.end())
    return it->second;
  wgpu::ComputePipelineDescriptor descriptor;
  descriptor.compute.module = shader;
  descriptor.compute.entryPoint = entryPoint;
  return kernels_[key] = device_.CreateComputePipeline(&descriptor);
}

wgpu::BindGroup Device::CreateBindGroup(
    const wgpu::ComputePipeline& kernel,
    std::initializer_list<wgpu::Buffer> buffers) {
  std::vector<wgpu::BindGroupEntry> entries;
  uint32_t index = 0;
  for (const wgpu::Buffer& buffer : buffers) {
    wgpu::BindGroupEntry entry;
    entry.binding = index++;
    entry.buffer = buffer;
    entries.push_back(std::move(entry));
  }
  wgpu::BindGroupDescriptor descriptor;
  descriptor.layout = kernel.GetBindGroupLayout(0);
  descriptor.entryCount = entries.size();
  descriptor.entries = entries.data();
  return device_.CreateBindGroup(&descriptor);
}

void Device::RunKernel(const wgpu::ComputePipeline& kernel,
                       const wgpu::BindGroup& bindGroup,
                       GridDims gridDims) {
  EnsureEncoder();
  wgpu::ComputePassEncoder pass = encoder_.BeginComputePass();
  pass.SetPipeline(kernel);
  pass.SetBindGroup(0, bindGroup);
  pass.DispatchWorkgroups(gridDims.x, gridDims.y, gridDims.z);
  pass.End();
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

void Device::AddFuture(const wgpu::Future& future) {
  {
    std::lock_guard lock(mutex_);
    futures_.insert(future.id);
  }
  hold_.notify_one();
}

// static
void Device::PollingThread(Device* self) {
  while (true) {
    // Collect futures to wait.
    std::vector<wgpu::FutureWaitInfo> infos;
    {
      std::lock_guard lock(self->mutex_);
      for (uint64_t futureID : self->futures_)
        infos.push_back({futureID});
    }
    if (infos.empty()) {
      // If there is no futures, wait until we get one.
      std::unique_lock lock(self->mutex_);
      self->hold_.wait(lock, [&]() {
        return self->shutdown_ || !self->futures_.empty();
      });
    } else {
      // Wait for a future to resolve, and then process events.
      wgpu::WaitStatus status = self->instance_.WaitAny(infos.size(),
                                                        infos.data(),
                                                        UINT64_MAX);
      assert(status == wgpu::WaitStatus::Success);
      self->instance_.ProcessEvents();
      // Erase completed ones.
      std::lock_guard lock(self->mutex_);
      for (const wgpu::FutureWaitInfo& info : infos) {
        if (info.completed)
          self->futures_.erase(info.future.id);
      }
    }
    if (self->shutdown_)
      break;
  }
}

}  // namespace betann
