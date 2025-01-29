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
  wgpu::RequestAdapterOptions options;
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  wgpu::Future future = instance_.RequestAdapter(
      &options,
      wgpu::CallbackMode::WaitAnyOnly,
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
  wgpu::RequiredLimits requiredLimits;
  requiredLimits.limits.maxComputeInvocationsPerWorkgroup =
      512;  // used by 3d binary general kernel
  wgpu::DeviceDescriptor deviceDescriptor;
  deviceDescriptor.requiredLimits = &requiredLimits;
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
      wgpu::CallbackMode::WaitAnyOnly,
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
}

Device::~Device() = default;

wgpu::Future Device::OnSubmittedWorkDone(std::function<void()> cb) {
  return AddFuture(queue_.OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [cb = std::move(cb)](wgpu::QueueWorkDoneStatus status) {
        if (status != wgpu::QueueWorkDoneStatus::Success) {
          throw std::runtime_error(
              fmt::format("OnSubmittedWorkDone failed: {}",
                          static_cast<uint32_t>(status)));
        }
        cb();
      }));
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

void Device::WaitFor(const wgpu::Future& future) {
  wgpu::FutureWaitInfo info{future.id};
  instance_.WaitAny(1, &info, UINT64_MAX);
}

void Device::WaitAll() {
  auto futures = std::move(futures_);
  while (!futures.empty()) {
    std::vector<wgpu::FutureWaitInfo> infos;
    for (uint64_t futureID : futures)
      infos.push_back({futureID});
    instance_.WaitAny(infos.size(), infos.data(), UINT64_MAX);
    for (const wgpu::FutureWaitInfo& info : infos) {
      if (info.completed)
        futures.erase(info.future.id);
    }
  }
}

wgpu::Buffer Device::CreateBuffer(size_t size, wgpu::BufferUsage usage) {
  wgpu::BufferDescriptor descriptor;
  descriptor.usage = usage;
  descriptor.size = size;
  return device_.CreateBuffer(&descriptor);
}

wgpu::Buffer Device::CreateBufferFromData(const void* data,
                                          size_t size,
                                          wgpu::BufferUsage usage) {
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
  wgpu::Buffer staging = CreateBuffer(buffer.GetSize(),
                                      wgpu::BufferUsage::MapRead |
                                      wgpu::BufferUsage::CopyDst);
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

wgpu::Future Device::ReadStagingBuffer(
    const wgpu::Buffer& buffer,
    std::function<void(const void* data)> cb) {
  return AddFuture(buffer.MapAsync(
      wgpu::MapMode::Read,
      0,
      wgpu::kWholeMapSize,
      wgpu::CallbackMode::WaitAnyOnly,
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
  wgpu::ComputePipeline kernel = device_.CreateComputePipeline(&descriptor);
  kernel.SetLabel(entryPoint);
  return kernels_[std::move(key)] = std::move(kernel);
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
                       Dims3 workgroupsCount) {
  EnsureEncoder();
  wgpu::ComputePassEncoder pass = encoder_.BeginComputePass();
  pass.SetPipeline(kernel);
  pass.SetBindGroup(0, bindGroup);
  pass.DispatchWorkgroups(workgroupsCount.x,
                          workgroupsCount.y,
                          workgroupsCount.z);
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

wgpu::Future Device::AddFuture(const wgpu::Future& future) {
  futures_.insert(future.id);
  return future;
}

}  // namespace betann
