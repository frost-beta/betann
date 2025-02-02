#include "betann_tests.h"

class UnaryTests : public BetaNNTests {
 public:
  template<typename T, typename U>
  std::vector<T> RunUnaryOpsContiguous(
      const char* op,
      const std::vector<U>& inputs,
      betann::DataType outputDataType = betann::GetDataType<T>(),
      betann::DataType inputDataType = betann::GetDataType<U>()) {
    wgpu::Buffer out = device_.CreateBuffer(
        inputs.size() * SizeOf(outputDataType),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::UnaryOpContiguous(
        device_,
        op,
        outputDataType,
        out,
        inputDataType,
        device_.CreateBufferFromVector(inputs, inputDataType),
        inputs.size());
    device_.Flush();
    return ReadFromBuffer<T>(out, inputs.size());
  }
};

TEST_F(UnaryTests, SpecialTypes) {
  EXPECT_EQ(RunUnaryOpsContiguous<uint32_t>("negative",
                                            std::vector<uint32_t>{100}),
            (std::vector<uint32_t>{4294967196}));
  EXPECT_EQ(RunUnaryOpsContiguous<uint16_t>(
                "exp",
                std::vector<uint16_t>{
                  betann::Float32ToFloat16(0.89),
                  betann::Float32ToFloat16(0.64),
                },
                betann::DataType::f16),
            (std::vector<uint16_t>{
                betann::Float32ToFloat16(2.43555),
                betann::Float32ToFloat16(1.89648),
            }));
}

TEST_F(UnaryTests, Contiguous) {
  EXPECT_EQ(RunUnaryOpsContiguous<float>("sqrt",
                                         std::vector<float>{0.64, 0.49, 0.36}),
            (std::vector<float>{0.8, 0.7, 0.6}));
  auto a = RandomNumbers<int32_t>(100);
  std::vector<uint32_t> b;
  for (int32_t i : a) {
    b.push_back(std::sqrt(static_cast<float>(i)));
  }
  EXPECT_EQ(RunUnaryOpsContiguous<uint32_t>("sqrt", a), b);
}
