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

  template<typename T, typename I>
  std::vector<T> RunCopyGeneral(const std::vector<I>& src,
                                const std::vector<uint32_t>& srcShape,
                                const std::vector<uint32_t>& srcStrides) {
    uint32_t dstNumElements = betann::NumElements(srcShape);
    wgpu::Buffer dst = device_.CreateBuffer(
        dstNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    dst.SetLabel("destination");
    betann::CopyGeneral(device_,
                        betann::GetWgslDataType<T>(),
                            dst,
                            betann::GetWgslDataType<I>(),
                            device_.CreateBufferFromVector(src),
                            srcShape,
                            srcStrides);
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

template<typename T>
std::vector<T> Duplicate(const std::vector<T>& vec, size_t times) {
  std::vector<T> ret;
  for (size_t i = 0; i < times; ++i)
    ret.insert(ret.end(), vec.begin(), vec.end());
  return ret;
}

TEST_F(CopyTests, SmallArrays) {
  auto src = RandomNumbers<int32_t>(100);
  auto dst = RunCopyContiguous<int32_t>(betann::CopyType::Vector,
                                        src.size(),
                                        src);
  EXPECT_EQ(dst, src);
  dst = RunCopyContiguous<int32_t>(betann::CopyType::Scalar,
                                   src.size(),
                                   src);
  EXPECT_EQ(dst, std::vector<int32_t>(src.size(), src[0]));
  src = RandomNumbers<int32_t>(100);
  auto dstFloat = RunCopyContiguous<float>(betann::CopyType::Vector,
                                           src.size(),
                                           src);
  EXPECT_EQ(dstFloat, std::vector<float>(src.begin(), src.end()));
}

TEST_F(CopyTests, LargeArrays) {
  uint32_t numElements =
      device_.GetLimits().maxComputeWorkgroupsPerDimension * 256 + 100;
  auto src = RandomNumbers<int32_t>(numElements);
  auto dst = RunCopyContiguous<int32_t>(betann::CopyType::Vector,
                                        numElements,
                                        src);
  EXPECT_EQ(dst, src);
}

TEST_F(CopyTests, GeneralContiguous) {
  auto src = RandomNumbers<int32_t>(64);
  auto dst = RunCopyGeneral<int32_t>(src, {4, 4, 4}, {16, 4, 1});
  EXPECT_EQ(dst, src);
}

TEST_F(CopyTests, GeneralNonContiguous) {
  // broadcast from 1x1 to 4x4x4
  std::vector<int32_t> src = {89};
  auto dst = RunCopyGeneral<int32_t>(src, {4, 4, 4}, {0, 0, 0});
  EXPECT_EQ(dst, std::vector<int32_t>(64, 89));
  // boradcast from 1x300 to 4x300
  src = RandomNumbers<int32_t>(300);
  dst = RunCopyGeneral<int32_t>(src, {4, 300}, {0, 1});
  EXPECT_EQ(dst, Duplicate(src, 4));
}

TEST_F(CopyTests, General4D) {
  // broadcast from 1x1 to 9x9x9x9
  std::vector<int32_t> src = {89};
  auto dst = RunCopyGeneral<int32_t>(src, {9, 9, 9, 9}, {0, 0, 0});
  EXPECT_EQ(dst, std::vector<int32_t>(9 * 9 * 9 * 9, 89));
  // boradcast from 1x100 to 9x9x9x100
  src = RandomNumbers<int32_t>(100);
  dst = RunCopyGeneral<int32_t>(src, {9, 9, 9, 100}, {0, 0, 0, 1});
  EXPECT_EQ(dst, Duplicate(src, 9 * 9 * 9));
  // boradcast from 1x100 to 1x2x3x4x100
  src = RandomNumbers<int32_t>(100);
  dst = RunCopyGeneral<int32_t>(src, {1, 2, 3, 4, 100}, {0, 0, 0, 0, 1});
  EXPECT_EQ(dst, Duplicate(src, 1 * 2 * 3 * 4));
}
