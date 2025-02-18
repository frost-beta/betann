#include "betann/device.h"

#include <array>
#include <stdexcept>

#include <fmt/format.h>

namespace betann {

Device::Device() {
  // Create instance.
  wgpu::InstanceDescriptor instanceDescriptor;
  instanceDescriptor.capabilities.timedWaitAnyEnable = true;
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
  supportsF16_ = adapter_.HasFeature(wgpu::FeatureName::ShaderF16);
  supportsSubgroups_ = adapter_.HasFeature(wgpu::FeatureName::Subgroups);
  supportsSubgroupsF16_ = adapter_.HasFeature(wgpu::FeatureName::SubgroupsF16);

  // Toggles for device.
  std::array toggles = {
    // Throw error immediately instead of delaying to next tick.
    "enable_immediate_error_handling",
    // Improve performance.
    "disable_lazy_clear_for_mapped_at_creation_buffer",
    "disable_robustness",
    "skip_validation",
  };
  wgpu::DawnTogglesDescriptor togglesDescriptor;
  togglesDescriptor.enabledToggles = toggles.data();
  togglesDescriptor.enabledToggleCount = toggles.size();
  // Limits for device.
  wgpu::RequiredLimits requiredLimits;
  requiredLimits.limits.maxComputeInvocationsPerWorkgroup =
      512;  // used by general kernels
  // Features for device.
  std::vector<wgpu::FeatureName> requiredFeatures;
  if (supportsF16_)
    requiredFeatures.push_back(wgpu::FeatureName::ShaderF16);
  if (supportsSubgroups_)
    requiredFeatures.push_back(wgpu::FeatureName::Subgroups);
  if (supportsSubgroupsF16_)
    requiredFeatures.push_back(wgpu::FeatureName::SubgroupsF16);

  // Synchronously request the device.
  wgpu::DeviceDescriptor deviceDescriptor;
  deviceDescriptor.nextInChain = &togglesDescriptor;
  deviceDescriptor.requiredLimits = &requiredLimits;
  deviceDescriptor.requiredFeatures = requiredFeatures.data();
  deviceDescriptor.requiredFeatureCount = requiredFeatures.size();
  deviceDescriptor.label = "BetaNN Device";
  deviceDescriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device& device,
         wgpu::DeviceLostReason reason,
         wgpu::StringView message) {
        if (reason != wgpu::DeviceLostReason::Destroyed) {
          throw std::runtime_error(
              fmt::format("Device lost: {}", message.data));
        }
      });
  deviceDescriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device& device,
         wgpu::ErrorType type,
         wgpu::StringView message) {
        throw std::runtime_error(message.data);
      });
  future = adapter_.RequestDevice(
      &deviceDescriptor,
      wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestDeviceStatus status,
             wgpu::Device result,
             wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          throw std::runtime_error(
              fmt::format("RequestDevice failed: {}", message.data));
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
    auto it = futures.begin();
    for (size_t i = 0; i < DivCeil(futures.size(), 64u); ++i) {
      std::vector<wgpu::FutureWaitInfo> infos;
      for (size_t j = 0; j < 64 && it != futures.end(); ++j, ++it) {
        infos.push_back({*it});
      }
      instance_.WaitAny(infos.size(), infos.data(), UINT64_MAX);
      for (const wgpu::FutureWaitInfo& info : infos) {
        if (info.completed)
          futures.erase(info.future.id);
      }
    }
  }
}

Buffer Device::CreateBuffer(uint64_t size,
                            BufferUsage usage,
                            bool mappedAtCreation) {
  wgpu::BufferDescriptor descriptor;
  descriptor.usage = usage;
  descriptor.size = size;
  descriptor.mappedAtCreation = mappedAtCreation;
  return {device_.CreateBuffer(&descriptor)};
}

Buffer Device::CreateBufferFromData(const void* data,
                                    uint64_t size,
                                    BufferUsage usage) {
  Buffer buffer = CreateBuffer(size, usage, true);
  memcpy(buffer.data.GetMappedRange(), data, size);
  buffer.data.Unmap();
  return buffer;
}

void Device::WriteBuffer(void* data, uint64_t size, Buffer& buffer) {
  queue_.WriteBuffer(buffer.data, buffer.offset, data, size);
}

wgpu::Future Device::ReadBuffer(const Buffer& buffer, ReadBufferCallback cb) {
  // Merge simultaneous read.
  WGPUBuffer key = buffer.data.Get();
  auto it = pendingReadBuffers_.find(key);
  if (it != pendingReadBuffers_.end()) {
    it->second.second.push_back(std::move(cb));
    return it->second.first;
  }
  // Create a staging buffer and copy to it.
  Buffer staging = CopyToStagingBuffer(buffer);
  Flush();
  // Map the buffer and read.
  wgpu::Future future = AddFuture(staging.data.MapAsync(
      wgpu::MapMode::Read,
      0,
      wgpu::kWholeMapSize,
      wgpu::CallbackMode::WaitAnyOnly,
      [this, staging, key](wgpu::MapAsyncStatus status, const char* message) {
        if (status != wgpu::MapAsyncStatus::Success)
          throw std::runtime_error(fmt::format("MapAsync failed: {}", message));
        // Invoke all the callbacks on the buffer.
        const void* stagingData = staging.data.GetConstMappedRange();
        for (const auto& cb : pendingReadBuffers_[key].second) {
          cb(stagingData, staging.GetSize(), staging.offset);
        }
        // Cleanup.
        staging.data.Unmap();
        pendingReadBuffers_.erase(key);
      }));
  pendingReadBuffers_[key] = {future, std::vector{std::move(cb)}};
  return future;
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

wgpu::BindGroup Device::CreateBindGroup(const wgpu::ComputePipeline& kernel,
                                        std::vector<Buffer> buffers) {
  std::vector<wgpu::BindGroupEntry> entries;
  uint32_t index = 0;
  for (Buffer& buffer : buffers) {
    if (buffer.data) {
      wgpu::BindGroupEntry entry;
      entry.binding = index++;
      entry.buffer = std::move(buffer.data);
      entry.size = buffer.size;
      entry.offset = buffer.offset;
      entries.push_back(std::move(entry));
    }
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

Buffer Device::CopyToStagingBuffer(const Buffer& buffer) {
  uint64_t totalSize = buffer.GetSize();
  Buffer staging = CreateBuffer(totalSize, BufferUsage::MapRead |
                                           BufferUsage::CopyDst);
  staging.size = buffer.size;
  staging.offset = buffer.offset;
  EnsureEncoder();
  encoder_.CopyBufferToBuffer(buffer.data, 0, staging.data, 0, totalSize);
  return staging;
}

wgpu::Future Device::AddFuture(const wgpu::Future& future) {
  futures_.insert(future.id);
  return future;
}

}  // namespace betann
