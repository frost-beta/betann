#include "betann_tests.h"

class ArangeTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> RunArange(
      double start,
      double step,
      uint32_t outNumElements,
      betann::DataType dataType = betann::GetDataType<T>()) {
    wgpu::Buffer out = device_.CreateBuffer(
        outNumElements * SizeOf(dataType),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::ArrayRange(device_,
                       start,
                       step,
                       dataType,
                       out);
    device_.Flush();
    return ReadFromBuffer<T>(out, outNumElements);
  }
};

TEST_F(ArangeTests, Sequence) {
  EXPECT_EQ(RunArange<uint32_t>(0, 1, 10), Iota<uint32_t>(10, 0));
  EXPECT_EQ(RunArange<int32_t>(-100, 1, 200), Iota<int32_t>(200, -100));
}
