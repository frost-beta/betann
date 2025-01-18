#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>

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
    wgpu::Buffer buf = device_.CreateBufferFromData(wgpu::BufferUsage::Storage,
                                                    vec.size() * sizeof(T),
                                                    vec.data());
    buf.SetLabel("input");
    return buf;
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

  template<typename T, typename I>
  std::vector<T> RunBinaryOp(betann::BinaryOpType type,
                             const char* name,
                             const std::vector<I>& a,
                             const std::vector<I>& b) {
    size_t outputSize = std::max(a.size(), b.size());
    wgpu::Buffer output = device_.CreateBuffer(
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        outputSize * sizeof(T));
    output.SetLabel("output");
    betann::BinaryOp(device_,
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

  template<typename T, typename I>
  std::vector<T> RunBinaryOpGeneral(const char* name,
                                    const std::vector<uint32_t>& shape,
                                    const std::vector<I>& a,
                                    const std::vector<uint32_t>& aStrides,
                                    const std::vector<I>& b,
                                    const std::vector<uint32_t>& bStrides) {
    size_t outputSize = std::accumulate(shape.begin(), shape.end(),
                                        1,
                                        std::multiplies<uint32_t>());
    wgpu::Buffer output = device_.CreateBuffer(
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        outputSize * sizeof(T));
    output.SetLabel("output");
    betann::BinaryOpGeneral(device_,
                            name,
                            shape,
                            betann::GetWgslDataType<T>(),
                            output,
                            betann::GetWgslDataType<I>(),
                            CreateBufferFromVector<I>(a),
                            a.size(),
                            aStrides,
                            CreateBufferFromVector<I>(b),
                            b.size(),
                            bStrides);
    device_.Flush();
    return ReadFromBuffer<T>(output, outputSize);
  }

  betann::Device device_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

TEST_F(BinaryTest, SmallArrays) {
  std::vector<float> ss = RunBinaryOp<float, uint32_t>(
      betann::BinaryOpType::ScalarScalar,
      "add",
      {8900},
      {64});
  EXPECT_EQ(ss, std::vector<float>({8964}));
  std::vector<int32_t> sv = RunBinaryOp<int32_t, int32_t>(
      betann::BinaryOpType::ScalarVector,
      "power",
      {2},
      {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(sv, std::vector<int32_t>({2, 4, 8, 16, 32, 64}));
  std::vector<int32_t> vs = RunBinaryOp<int32_t, float>(
      betann::BinaryOpType::VectorScalar,
      "multiply",
      {64, 64, 64},
      {140.0625});
  EXPECT_EQ(vs, std::vector<int32_t>({8964, 8964, 8964}));
  std::vector<float> vv = RunBinaryOp<float, float>(
      betann::BinaryOpType::VectorVector,
      "subtract",
      {100, 100, 100},
      {10.36, 10.36, 10.36});
  EXPECT_EQ(vv, std::vector<float>({89.64, 89.64, 89.64}));
}

TEST_F(BinaryTest, LargeArrays) {
  uint32_t outputSize =
      device_.GetLimits().maxComputeWorkgroupsPerDimension + 100;
  std::vector<uint32_t> a(outputSize);
  std::fill(a.begin(), a.end(), 8900);
  std::vector<uint32_t> b(outputSize);
  std::fill(b.begin(), b.end(), 64);
  std::vector<float> vv = RunBinaryOp<float, uint32_t>(
      betann::BinaryOpType::VectorVector,
      "add",
      a,
      b);
  EXPECT_TRUE(std::all_of(vv.begin(), vv.end(),
                          [](float i) { return i == 8964; }));
}

TEST_F(BinaryTest, GeneralContiguous) {
  std::vector<float> con1d = RunBinaryOpGeneral<float, float>(
      "subtract",
      {3},
      {1, 2, 3},
      {1},
      {0.1, 0.2, 0.3},
      {1});
  EXPECT_EQ(con1d, std::vector<float>({0.9, 1.8, 2.7}));
  std::vector<int32_t> con2d = RunBinaryOpGeneral<int32_t, int32_t>(
      "multiply",
      {2, 2},
      {1, 2, 3, 4},
      {2, 1},
      {2, 2, 2, 2},
      {2, 1});
  EXPECT_EQ(con2d, std::vector<int32_t>({2, 4, 6, 8}));
  std::vector<uint32_t> con3d = RunBinaryOpGeneral<uint32_t, uint32_t>(
      "subtract",
      {1, 3, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9},
      {9, 3, 1},
      {5, 5, 5, 5, 5, 5, 5, 5, 5},
      {9, 3, 1});
  uint32_t m = static_cast<uint32_t>(-1);
  EXPECT_EQ(
      con3d,
      std::vector<uint32_t>({m - 3, m - 2, m - 1, m, 0, 1, 2, 3, 4}));
}

TEST_F(BinaryTest, GeneralNonContiguous) {
  std::vector<float> noc1d = RunBinaryOpGeneral<float, float>(
      "multiply",
      {2},
      {1, 2, 3, 4},
      {2},
      {2},
      {1});
  EXPECT_EQ(noc1d, std::vector<float>({2, 6}));
  std::vector<float> noc2d = RunBinaryOpGeneral<float, float>(
      "add",
      {2, 1},
      {1, 2, 3, 4},
      {2, 1},
      {2},
      {0, 0});
  EXPECT_EQ(noc2d, std::vector<float>({3, 5}));
  std::vector<float> noc3d = RunBinaryOpGeneral<float, float>(
      "divide",
      {2, 2, 2},
      {2, 4, 6, 8, 10, 12, 14, 16},
      {0, 1, 2},
      {2},
      {0, 0});
  EXPECT_EQ(noc3d, std::vector<float>({1, 3, 2, 4, 1, 3, 2, 4}));
}
