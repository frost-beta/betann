#include "betann_tests.h"

class BinaryTests : public BetaNNTests {
 public:
  template<typename T, typename I>
  std::vector<T> RunBinaryOpContiguous(betann::BinaryOpType type,
                                       const char* name,
                                       const std::vector<I>& a,
                                       const std::vector<I>& b) {
    size_t outputNumElements = std::max(a.size(), b.size());
    betann::Buffer output = device_.CreateBuffer(
        outputNumElements * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    output.data.SetLabel("output");
    betann::BinaryOpContiguous(device_,
                               name,
                               type,
                               betann::GetDataType<T>(),
                               output,
                               outputNumElements,
                               betann::GetDataType<I>(),
                               device_.CreateBufferFromVector(a),
                               device_.CreateBufferFromVector(b));
    device_.Flush();
    return ReadFromBuffer<T>(output, outputNumElements);
  }

  template<typename T, typename I>
  std::vector<T> RunBinaryOpGeneral(const char* name,
                                    const std::vector<uint32_t>& shape,
                                    const std::vector<I>& a,
                                    const std::vector<uint32_t>& aStrides,
                                    const std::vector<I>& b,
                                    const std::vector<uint32_t>& bStrides) {
    uint32_t outputNumElements = betann::NumElements(shape);
    betann::Buffer output = device_.CreateBuffer(
        outputNumElements * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    output.data.SetLabel("output");
    betann::BinaryOpGeneral(device_,
                            name,
                            betann::GetDataType<T>(),
                            output,
                            shape,
                            betann::GetDataType<I>(),
                            device_.CreateBufferFromVector(a),
                            aStrides,
                            device_.CreateBufferFromVector(b),
                            bStrides);
    device_.Flush();
    return ReadFromBuffer<T>(output, outputNumElements);
  }
};

TEST_F(BinaryTests, SmallArrays) {
  std::vector<float> ss = RunBinaryOpContiguous<float, uint32_t>(
      betann::BinaryOpType::ScalarScalar,
      "add",
      {8900},
      {64});
  EXPECT_EQ(ss, std::vector<float>({8964}));
  std::vector<int32_t> sv = RunBinaryOpContiguous<int32_t, int32_t>(
      betann::BinaryOpType::ScalarVector,
      "power",
      {2},
      {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(sv, std::vector<int32_t>({2, 4, 8, 16, 32, 64}));
  std::vector<int32_t> vs = RunBinaryOpContiguous<int32_t, float>(
      betann::BinaryOpType::VectorScalar,
      "multiply",
      {64, 64, 64},
      {140.0625});
  EXPECT_EQ(vs, std::vector<int32_t>({8964, 8964, 8964}));
  std::vector<float> vv = RunBinaryOpContiguous<float, float>(
      betann::BinaryOpType::VectorVector,
      "subtract",
      {100, 100, 100},
      {10.36, 10.36, 10.36});
  EXPECT_EQ(vv, std::vector<float>({89.64, 89.64, 89.64}));
}

TEST_F(BinaryTests, LargeArrays) {
  uint32_t outputNumElements =
      device_.GetLimits().maxComputeWorkgroupsPerDimension * 64 + 100;
  std::vector<uint32_t> a(outputNumElements);
  std::fill(a.begin(), a.end(), 8900);
  std::vector<uint32_t> b(outputNumElements);
  std::fill(b.begin(), b.end(), 64);
  std::vector<float> vv = RunBinaryOpContiguous<float, uint32_t>(
      betann::BinaryOpType::VectorVector,
      "add",
      a,
      b);
  EXPECT_TRUE(std::all_of(vv.begin(), vv.end(),
                          [](float i) { return i == 8964; }));
}

TEST_F(BinaryTests, GeneralContiguous) {
  std::vector<float> con1d = RunBinaryOpGeneral<float, float>(
      "subtract",
      {1, 3},
      {1, 2, 3},
      {0, 1},
      {0.1, 0.2, 0.3},
      {0, 1});
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

TEST_F(BinaryTests, GeneralNonContiguous) {
  std::vector<float> noc1d = RunBinaryOpGeneral<float, float>(
      "multiply",
      {1, 2},
      {1, 2, 3, 4},
      {0, 2},
      {2},
      {0, 0});
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
      {0, 0, 0});
  EXPECT_EQ(noc3d, std::vector<float>({1, 3, 2, 4, 1, 3, 2, 4}));
}

TEST_F(BinaryTests, GeneralLargeArrays) {
  std::vector<uint32_t> shape = {33, 33, 33};
  std::vector<uint32_t> strides = {33 * 33, 33, 1};
  uint32_t outputNumElements = betann::NumElements(shape);
  std::vector<float> a(outputNumElements, 8);
  std::vector<float> b(outputNumElements, 8);
  auto c = RunBinaryOpGeneral<uint32_t, float>(
      "multiply", shape, a, strides, b, strides);
  EXPECT_EQ(c, std::vector<uint32_t>(c.size(), 64));
}

TEST_F(BinaryTests, General4D) {
  std::vector<uint32_t> row = RunBinaryOpGeneral<uint32_t, uint32_t>(
      "add",
      {1, 1, 1, 4},
      {1, 2, 3, 4},
      {4, 4, 4, 1},
      {7, 7, 3, 0},
      {4, 4, 4, 1});
  EXPECT_EQ(row, std::vector<uint32_t>({8, 9, 6, 4}));
  auto arange = Iota<uint32_t>(16, 1);
  std::vector<uint32_t> cube = RunBinaryOpGeneral<uint32_t, uint32_t>(
      "multiply",
      {2, 2, 2, 2},
      arange,
      {8, 4, 2, 1},
      std::vector<uint32_t>(16, 2),
      {8, 4, 2, 1});
  EXPECT_EQ(cube, Map(arange, [](uint32_t i) { return i * 2; }));
}

TEST_F(BinaryTests, General4DLargeArrays) {
  std::vector<uint32_t> shape = {33, 33, 33, 33};
  std::vector<uint32_t> strides = {33 * 33 * 33, 33 * 33, 33, 1};
  uint32_t outputNumElements = betann::NumElements(shape);
  auto a = Iota<float>(outputNumElements, 1);
  auto b = Iota<float>(outputNumElements, 1);
  auto c = RunBinaryOpGeneral<float, float>(
      "multiply", shape, a, strides, b, strides);
  EXPECT_EQ(c, Map(a, [](float i) { return i * i; }));
}
