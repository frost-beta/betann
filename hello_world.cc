#include <condition_variable>
#include <iostream>
#include <mutex>

#include <betann.h>

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

  uint32_t number = 89;
  wgpu::Buffer buffer = device.CreateBufferFromData(
      wgpu::BufferUsage::Storage |
      wgpu::BufferUsage::CopySrc |
      wgpu::BufferUsage::CopyDst,
      sizeof(number),
      &number);

  wgpu::ShaderModule shader = device.CreateShaderModule(
      "tiananmen",
      R"(
        @group(0) @binding(0) var<storage, read_write> ssbo : u32;
        @compute @workgroup_size(1) fn date() {
            ssbo *= 100;
            ssbo += 64;
        }
      )");
  wgpu::ComputePipeline kernel = device.CreateKernel(shader, "date");
  wgpu::BindGroup bindGroup = device.CreateBindGroup(kernel, {buffer});
  device.RunKernel(kernel, bindGroup, {1});
  device.Flush();

  wgpu::Buffer staging = device.CopyToStagingBuffer(buffer);
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
