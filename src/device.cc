#include "src/device.h"

#include <string>

namespace betann {

Device::Device() {
  wgpu::InstanceDescriptor instance_descriptor{};
  instance_descriptor.features.timedWaitAnyEnable = true;
  wgpu::Instance instance = wgpu::CreateInstance(&instance_descriptor);
  if (!instance)
    throw std::runtime_error("CreateInstance failed.");
  // Synchronously request the adapter.
  wgpu::RequestAdapterOptions options = {};
  wgpu::Future future = instance.RequestAdapter(
      &options,
      wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestAdapterStatus status,
             wgpu::Adapter result,
             const char* message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          std::string error = "RequestAdapter failed: ";
          throw std::runtime_error(error + message);
        }
        adapter_ = std::move(result);
      });
  instance.WaitAny(future, 5 * 1000);
  // Synchronously request the device.
  wgpu::DeviceDescriptor device_descriptor;
  device_descriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device& device,
         wgpu::DeviceLostReason reason,
         const char* message) {
        if (reason != wgpu::DeviceLostReason::Destroyed) {
          std::string error = "Device lost: ";
          throw std::runtime_error(error + message);
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
      wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestDeviceStatus status,
             wgpu::Device result,
             const char* message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::string error = "RequestDevice failed: ";
          throw std::runtime_error(error + message);
        }
        device_ = std::move(result);
      });
  instance.WaitAny(future, 5 * 1000);
}

Device::~Device() = default;

}  // namespace betann
