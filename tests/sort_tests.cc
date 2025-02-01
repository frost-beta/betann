#include "betann_tests.h"

class SortTests : public BetaNNTests {
 public:
  template<typename T>
  std::pair<std::vector<T>, std::vector<uint32_t>> RunSortContiguous(
      uint32_t axis,
      const std::vector<T>& input,
      const std::vector<uint32_t>& shape,
      const std::vector<uint32_t>& strides) {
    uint32_t inputNumElements = betann::NumElements(shape, strides);
    wgpu::Buffer out = device_.CreateBuffer(
        inputNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    wgpu::Buffer sortedIndices = device_.CreateBuffer(
        inputNumElements * sizeof(uint32_t),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    out.SetLabel("out");
    betann::SortSingleBlockContiguous(device_,
                                      axis,
                                      betann::GetDataType<T>(),
                                      out,
                                      sortedIndices,
                                      strides,
                                      device_.CreateBufferFromVector(input),
                                      shape,
                                      strides);
    device_.Flush();
    return {
      ReadFromBuffer<T>(out, inputNumElements),
      ReadFromBuffer<uint32_t>(sortedIndices, inputNumElements),
    };
  }

  template<typename T>
  std::vector<T> Sorted(std::vector<T> a) {
    std::sort(a.begin(), a.end());
    return a;
  }

  template<typename T>
  std::vector<T> Concat(std::vector<T> a, const std::vector<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
  }
};

TEST_F(SortTests, SingleBlockContiguous) {
  auto a = RandomNumbers<uint32_t>(100);
  EXPECT_EQ(RunSortContiguous<uint32_t>(0, a, {100}, {1}).first,
            Sorted(a));
  auto b = RandomNumbers<uint32_t>(100);
  auto c = Concat(a, b);
  EXPECT_EQ(RunSortContiguous<uint32_t>(1, c, {2, 100}, {100, 1}).first,
            Concat(Sorted(a), Sorted(b)));
  EXPECT_EQ(RunSortContiguous<uint32_t>(0, c, {200, 1}, {1, 0}).first,
            Sorted(c));
}

TEST_F(SortTests, ArgSortSingleBlockContiguous) {
  std::vector<float> a = {8, 9, 6, 4};
  EXPECT_EQ(RunSortContiguous<float>(0, a, {4}, {1}).second,
            (std::vector<uint32_t>{3, 2, 0, 1}));
  std::vector<float> b = {4, 3, 2, 1};
  auto c = Concat(a, b);
  EXPECT_EQ(RunSortContiguous<float>(1, c, {2, 4}, {4, 1}).second,
            (std::vector<uint32_t>{3, 2, 0, 1, 3, 2, 1, 0}));
}
