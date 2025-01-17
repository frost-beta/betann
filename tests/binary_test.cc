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

  template<typename T, typename I, typename Op>
  std::vector<T> RunOp(Op op,
                       betann::BinaryOpType type,
                       const char* name,
                       const std::vector<I>& a,
                       const std::vector<I>& b) {
    size_t outputSize = std::max(a.size(), b.size());
    wgpu::Buffer output = device_.CreateBuffer(
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        outputSize * sizeof(T));
    op(device_,
       type,
       name,
       outputSize,
       betann::GetWgslDataType<T>(),
       output,
       betann::GetWgslDataType<I>(),
       CreateBufferFromVector<I>(a),
       CreateBufferFromVector<I>(b));
    device_.Flush();
    return ReadFromBuffer<T>(output, outputSize);
  }

  betann::Device device_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

TEST_F(BinaryTest, SmallArrays) {
  std::vector<float> ss = RunOp<float, uint32_t>(
      betann::BinaryOp,
      betann::BinaryOpType::ScalarScalar,
      "add",
      {8900},
      {64});
  EXPECT_EQ(ss, std::vector<float>({8964}));
  std::vector<int32_t> sv = RunOp<int32_t, int32_t>(
      betann::BinaryOp,
      betann::BinaryOpType::ScalarVector,
      "power",
      {2},
      {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(sv, std::vector<int32_t>({2, 4, 8, 16, 32, 64}));
  std::vector<int32_t> vs = RunOp<int32_t, float>(
      betann::BinaryOp,
      betann::BinaryOpType::VectorScalar,
      "multiply",
      {64, 64, 64},
      {140.0625});
  EXPECT_EQ(vs, std::vector<int32_t>({8964, 8964, 8964}));
  std::vector<float> vv = RunOp<float, float>(
      betann::BinaryOp,
      betann::BinaryOpType::VectorVector,
      "subtract",
      {100, 100, 100},
      {10.36, 10.36, 10.36});
  EXPECT_EQ(vv, std::vector<float>({89.64, 89.64, 89.64}));
}
