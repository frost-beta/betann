#include "betann_tests.h"

class RandomTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> RunRandomBitsContiguous(uint32_t outNumElements,
                                         const std::vector<uint32_t>& keys) {
    wgpu::Buffer out = device_.CreateBuffer(
        outNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    out.SetLabel("out");
    betann::RandomBitsContiguous(device_,
                                 betann::GetDataType<T>(),
                                 out,
                                 outNumElements,
                                 device_.CreateBufferFromVector(keys),
                                 keys.size());
    device_.Flush();
    return ReadFromBuffer<T>(out, outNumElements);
  }

  template<typename T>
  std::vector<T> RunRandomBitsGeneral(uint32_t outNumElements,
                                      const std::vector<uint32_t>& keys,
                                      const std::vector<uint32_t>& shape,
                                      const std::vector<uint32_t>& strides) {
    wgpu::Buffer out = device_.CreateBuffer(
        outNumElements * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    out.SetLabel("out");
    betann::RandomBitsGeneral(device_,
                              betann::GetDataType<T>(),
                              out,
                              outNumElements,
                              device_.CreateBufferFromVector(keys),
                              shape,
                              strides);
    device_.Flush();
    return ReadFromBuffer<T>(out, outNumElements);
  }

  std::vector<uint32_t> Key(uint64_t seed) {
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    uint32_t k2 = static_cast<uint32_t>(seed);
    return {k1, k2};
  }
};

TEST_F(RandomTests, Contiguous) {
  EXPECT_EQ(RunRandomBitsContiguous<uint32_t>(1, Key(0)),
            std::vector<uint32_t>{1797259609});
  EXPECT_EQ(RunRandomBitsContiguous<uint32_t>(1, Key(1)),
            std::vector<uint32_t>{507451445});
  EXPECT_EQ(RunRandomBitsContiguous<uint32_t>(3, Key(0)),
            (std::vector<uint32_t>{4146024105, 1351547692, 2718843009}));
}

TEST_F(RandomTests, General) {
  EXPECT_EQ(RunRandomBitsGeneral<uint32_t>(1, {0}, {2}, {1}),
            std::vector<uint32_t>{1797259609});
  EXPECT_EQ(RunRandomBitsGeneral<uint32_t>(1, {0, 8, 1, 8}, {2}, {2}),
            std::vector<uint32_t>{507451445});
  EXPECT_EQ(RunRandomBitsGeneral<uint32_t>(3, {0, 8, 0, 8}, {2}, {2}),
            (std::vector<uint32_t>{4146024105, 1351547692, 2718843009}));
}
