#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>

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
  std::string data = "Hello World";

  wgpu::Buffer buffer = device.CreateBuffer(wgpu::BufferUsage::Storage |
                                            wgpu::BufferUsage::CopySrc |
                                            wgpu::BufferUsage::CopyDst,
                                            data.size() + 1,
                                            data.data());
  wgpu::Buffer staging = device.CopyToStagingBuffer(buffer);
  device.Flush();
  device.ReadStagingBuffer(
      staging,
      [&](const void* data) {
        std::cerr << static_cast<const char*>(data) << std::endl;
        wakeup();
      });
  wait();

  return 0;
}
