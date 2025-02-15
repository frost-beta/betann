#include "betann_tests.h"

class UnaryTests : public BetaNNTests {
 public:
  template<typename T, typename U>
  std::vector<T> RunUnaryOpsContiguous(
      const char* op,
      const std::vector<U>& input,
      betann::DataType outputDataType = betann::GetDataType<T>(),
      betann::DataType inputDataType = betann::GetDataType<U>()) {
    betann::Buffer out = device_.CreateBuffer(
        input.size() * SizeOf(outputDataType),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::UnaryOpContiguous(
        device_,
        op,
        outputDataType,
        out,
        inputDataType,
        device_.CreateBufferFromVector(input, inputDataType),
        input.size());
    device_.Flush();
    return ReadFromBuffer<T>(out, input.size());
  }

  template<typename T, typename U>
  std::vector<T> RunUnaryOpsGeneral(
      const char* op,
      const std::vector<U>& input,
      const std::vector<uint32_t>& inputShape,
      const std::vector<uint32_t>& inputStrides,
      betann::DataType outputDataType = betann::GetDataType<T>(),
      betann::DataType inputDataType = betann::GetDataType<U>()) {
    betann::Buffer out = device_.CreateBuffer(
        betann::NumElements(inputShape, inputStrides) * SizeOf(outputDataType),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::UnaryOpGeneral(
        device_,
        op,
        outputDataType,
        out,
        inputDataType,
        device_.CreateBufferFromVector(input, inputDataType),
        inputShape,
        inputStrides);
    device_.Flush();
    return ReadFromBuffer<T>(out, input.size());
  }
};

TEST_F(UnaryTests, SpecialTypes) {
  EXPECT_EQ(RunUnaryOpsContiguous<uint32_t>("negative",
                                            std::vector<uint32_t>{100}),
            (std::vector<uint32_t>{4294967196}));
  if (device_.SupportsF16()) {
    EXPECT_EQ(RunUnaryOpsContiguous<uint16_t>(
                  "exp",
                  std::vector<uint16_t>{
                    betann::Float32ToFloat16(0.89),
                    betann::Float32ToFloat16(0.64),
                  },
                  betann::DataType::F16),
              (std::vector<uint16_t>{
                  betann::Float32ToFloat16(2.43555),
                  betann::Float32ToFloat16(1.89648),
              }));
  }
}

TEST_F(UnaryTests, Contiguous) {
  EXPECT_EQ(RunUnaryOpsContiguous<int32_t>("sqrt",
                                           std::vector<int32_t>{64, 49, 36}),
            (std::vector<int32_t>{8, 7, 6}));
  auto a = RandomNumbers<int32_t>(1000);
  EXPECT_EQ(RunUnaryOpsContiguous<int32_t>("sqrt", a),
            Map(a, [](int32_t i) {
              return static_cast<int32_t>(std::sqrt(static_cast<float>(i)));
            }));
}

TEST_F(UnaryTests, General) {
  auto a = RandomNumbers<int32_t>(1000);
  EXPECT_EQ(RunUnaryOpsGeneral<int32_t>("negative",
                                        a,
                                        {10, 10, 10},
                                        {100, 10, 1}),
            Map(a, [](int32_t i) { return -i; }));
  EXPECT_EQ(RunUnaryOpsGeneral<int32_t>("square",
                                        a,
                                        {2, 5, 10, 10},
                                        {500, 100, 10, 1}),
            Map(a, [](int32_t i) { return i * i; }));
  EXPECT_THROW({
    RunUnaryOpsGeneral<int32_t>("negative", a, {1000}, {1});
  }, std::runtime_error);
}
