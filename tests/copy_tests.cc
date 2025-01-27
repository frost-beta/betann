#include "betann_tests.h"

#include <random>

class CopyTests : public BetaNNTests {
 public:
  template<typename T, typename I>
  std::vector<T> RunCopyContiguous(betann::CopyType type,
                                   size_t dstNumElements,
                                   const std::vector<I>& src) {
    wgpu::Buffer dst = device_.CreateBuffer(
        dstNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    dst.SetLabel("destination");
    betann::CopyContiguous(device_,
                           type,
                           betann::GetWgslDataType<T>(),
                           dst,
                           dstNumElements,
                           betann::GetWgslDataType<I>(),
                           device_.CreateBufferFromVector(src));
    device_.Flush();
    return ReadFromBuffer<T>(dst, dstNumElements);
  }

  template<typename T>
  std::vector<T> RandomNumbers(size_t size) {
    std::uniform_int_distribution<T> dist(0, 8964);
    std::vector<T> ret;
    for (size_t i = 0; i < size; ++i)
      ret.push_back(dist(mt_));
    return ret;
  }

 private:
  std::mt19937 mt_;
};

TEST_F(CopyTests, SmallArrays) {
  auto src = RandomNumbers<int32_t>(100);
  auto dst = RunCopyContiguous<int32_t>(betann::CopyType::Vector,
                                        src.size(),
                                        src);
  EXPECT_EQ(src, dst);
  dst = RunCopyContiguous<int32_t>(betann::CopyType::Scalar,
                                   src.size(),
                                   src);
  EXPECT_EQ(src.size(), dst.size());
  std::fill(src.begin(), src.end(), src[0]);
  EXPECT_EQ(src, dst);
  src = RandomNumbers<int32_t>(100);
  auto dstFloat = RunCopyContiguous<float>(betann::CopyType::Vector,
                                           src.size(),
                                           src);
  auto srcFloat = std::vector<float>(src.begin(), src.end());
  EXPECT_EQ(srcFloat, dstFloat);
}

TEST_F(CopyTests, LargeArrays) {
  uint32_t numElements =
      device_.GetLimits().maxComputeWorkgroupsPerDimension * 3 + 100;
  auto src = RandomNumbers<int32_t>(numElements);
  auto dst = RunCopyContiguous<int32_t>(betann::CopyType::Vector,
                                        numElements,
                                        src);
  EXPECT_EQ(src, dst);
}
