#include "betann_tests.h"

class SortTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> RunSortSingleBlockContiguous(
      uint32_t axis,
      const std::vector<T>& input,
      const std::vector<uint32_t>& shape,
      const std::vector<uint32_t>& strides) {
    uint32_t inputNumElements = betann::NumElements(shape, strides);
    wgpu::Buffer out = device_.CreateBuffer(
        inputNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    out.SetLabel("out");
    betann::SortSingleBlockContiguous(device_,
                                      axis,
                                      betann::GetDataType<T>(),
                                      out,
                                      strides,
                                      device_.CreateBufferFromVector(input),
                                      shape,
                                      strides);
    device_.Flush();
    return ReadFromBuffer<T>(out, inputNumElements);
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
  EXPECT_EQ(RunSortSingleBlockContiguous<uint32_t>(0, a, {100}, {1}),
            Sorted(a));
  auto b = RandomNumbers<uint32_t>(100);
  auto c = Concat(a, b);
  EXPECT_EQ(RunSortSingleBlockContiguous<uint32_t>(1, c, {2, 100}, {100, 1}),
            Concat(Sorted(a), Sorted(b)));
  EXPECT_EQ(RunSortSingleBlockContiguous<uint32_t>(0, c, {200, 1}, {1, 0}),
            Sorted(c));
}
