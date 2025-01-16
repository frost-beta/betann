#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>

#include <betann/betann.h>
#include <gtest/gtest.h>

class BinaryTest : public testing::Test {
 protected:
  void Wakeup() {
    std::lock_guard lock(mutex_);
    cv_.notify_one();
  };

  void Wait() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock);
  };

  template<typename T>
  wgpu::Buffer CreateBufferFromVector(const std::vector<T>& vec) {
    return device_.CreateBufferFromData(wgpu::BufferUsage::Storage,
                                        vec.size() * sizeof(T),
                                        vec.data());
  }

  template<typename T>
  std::vector<T> ReadFromBuffer(const wgpu::Buffer& buf, size_t size) {
    wgpu::Buffer staging = device_.CopyToStagingBuffer(buf);
    device_.Flush();
    std::vector<T> out(size);
    device_.ReadStagingBuffer(
        staging,
        [&](const void* data) {
          std::memcpy(out.data(), data, size * sizeof(T));
          Wakeup();
        });
    Wait();
    return out;
  }

  betann::Device device_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

TEST_F(BinaryTest, Add) {
  wgpu::Buffer output = device_.CreateBuffer(
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
      sizeof(float));
  betann::BinaryOp(device_,
                   betann::BinaryOpType::ScalarScalar,
                   "add",
                   1,
                   "f32",
                   output,
                   "u32",
                   CreateBufferFromVector<uint32_t>({8900}),
                   CreateBufferFromVector<uint32_t>({64}));
  device_.Flush();
  std::vector<float> c = ReadFromBuffer<float>(output, 1);
  ASSERT_EQ(c[0], 8964);
}
