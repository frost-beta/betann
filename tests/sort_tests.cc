#include "betann_tests.h"

class SortTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> RunSort(uint32_t axis,
                         betann::SortInputType inputType,
                         const std::vector<T>& input,
                         const std::vector<uint32_t>& shape,
                         const std::vector<uint32_t>& strides) {
    uint32_t inputNumElements = betann::NumElements(shape, strides);
    wgpu::Buffer out = device_.CreateBuffer(
        inputNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::SortBlock(device_,
                      axis,
                      inputType,
                      betann::SortResultType::Values,
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
  std::vector<uint32_t> RunArgSort(uint32_t axis,
                                   betann::SortInputType inputType,
                                   const std::vector<T>& input,
                                   const std::vector<uint32_t>& shape,
                                   const std::vector<uint32_t>& strides) {
    uint32_t inputNumElements = betann::NumElements(shape, strides);
    wgpu::Buffer sortedIndices = device_.CreateBuffer(
        inputNumElements * sizeof(uint32_t),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::SortBlock(device_,
                      axis,
                      inputType,
                      betann::SortResultType::Indices,
                      betann::GetDataType<T>(),
                      sortedIndices,
                      strides,
                      device_.CreateBufferFromVector(input),
                      shape,
                      strides);
    device_.Flush();
    return ReadFromBuffer<uint32_t>(sortedIndices, inputNumElements);
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
  for (auto type : {betann::SortInputType::Contiguous,
                    betann::SortInputType::General}) {
    auto a = RandomNumbers<uint32_t>(100);
    EXPECT_EQ(RunSort<uint32_t>(0, type, a, {100}, {1}),
              Sorted(a));
    auto b = RandomNumbers<uint32_t>(100);
    auto c = Concat(a, b);
    EXPECT_EQ(RunSort<uint32_t>(1, type, c, {2, 100}, {100, 1}),
              Concat(Sorted(a), Sorted(b)));
    EXPECT_EQ(RunSort<uint32_t>(0, type, c, {200, 1}, {1, 0}),
              Sorted(c));
  }
}

TEST_F(SortTests, SingleBlockGeneral) {
  std::vector<int32_t> a(24);
  std::iota(a.begin(), a.end(), 1);
  EXPECT_EQ(RunSort<int32_t>(1,
                             betann::SortInputType::General,
                             a,
                             {2, 3, 4},
                             {12, 4, 1}),
            a);
  std::reverse(a.begin(), a.end());
  EXPECT_EQ(RunSort<int32_t>(1,
                             betann::SortInputType::General,
                             a,
                             {2, 3, 4},
                             {12, 4, 1}),
            (std::vector<int32_t>{16, 15, 14, 13, 20, 19, 18, 17,
                                  24, 23, 22, 21, 4,  3,  2,  1,
                                  8,  7,  6,  5,  12, 11, 10, 9}));
}

TEST_F(SortTests, ArgSortSingleBlockContiguous) {
  for (auto type : {betann::SortInputType::Contiguous,
                    betann::SortInputType::General}) {
    std::vector<float> a = {8, 9, 6, 4};
    EXPECT_EQ(RunArgSort<float>(0, type, a, {4}, {1}),
              (std::vector<uint32_t>{3, 2, 0, 1}));
    std::vector<float> b = {4, 3, 2, 1};
    auto c = Concat(a, b);
    EXPECT_EQ(RunArgSort<float>(1, type, c, {2, 4}, {4, 1}),
              (std::vector<uint32_t>{3, 2, 0, 1, 3, 2, 1, 0}));
  }
}
