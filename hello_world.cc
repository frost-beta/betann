#include <condition_variable>
#include <iostream>
#include <mutex>

#include "betann/betann.h"

int main(int argc, char *argv[]) {
  std::mutex mutex;
  std::condition_variable cv;
  auto wakeup = [&]() {
    std::lock_guard lock(mutex);
    cv.notify_one();
  };
  auto wait = [&]() {
    std::unique_lock lock(mutex);
    cv.wait(lock);
  };

  betann::Device device;

  uint32_t number = 8900;
  wgpu::Buffer a = device.CreateBufferFromData(
      wgpu::BufferUsage::Storage |
      wgpu::BufferUsage::CopySrc |
      wgpu::BufferUsage::CopyDst,
      sizeof(number),
      &number);
  number = 64;
  wgpu::Buffer b = device.CreateBufferFromData(
      wgpu::BufferUsage::Storage |
      wgpu::BufferUsage::CopySrc |
      wgpu::BufferUsage::CopyDst,
      sizeof(number),
      &number);
  wgpu::Buffer c = device.CreateBuffer(
      wgpu::BufferUsage::Storage |
      wgpu::BufferUsage::CopySrc |
      wgpu::BufferUsage::CopyDst,
      sizeof(number));

  wgpu::ShaderModule shader = device.CreateShaderModule(
      "tiananmen",
      betann::GetBinaryShaderSource("add", "u32", "u32").c_str());
  wgpu::ComputePipeline kernel = device.CreateKernel(shader, "binary_ss_add");
  wgpu::BindGroup bindGroup = device.CreateBindGroup(kernel, {a, b, c});
  device.RunKernel(kernel, bindGroup, {1});
  device.Flush();

  wgpu::Buffer staging = device.CopyToStagingBuffer(c);
  device.Flush();
  device.ReadStagingBuffer(
      staging,
      [&](const void* data) {
        std::cerr << *static_cast<const uint32_t*>(data) << std::endl;
        wakeup();
      });
  wait();

  return 0;
}
