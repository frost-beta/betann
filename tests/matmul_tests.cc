#include "betann_tests.h"

class MatrixMultiplyTest : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> GpuMatmul(const std::vector<T>& a,
                           const std::vector<uint32_t>& aShape,
                           const std::vector<uint32_t>& aStrides,
                           const std::vector<T>& b,
                           const std::vector<uint32_t>& bShape,
                           const std::vector<uint32_t>& bStrides) {
    uint32_t outSize = aShape[aShape.size() - 2] * bShape[bShape.size() - 1];
    if (aShape.size() > 2) {
      for (size_t i = 0; i < aShape.size() - 2; ++i)
        outSize *= aShape[i];
    }
    betann::Buffer out = device_.CreateBuffer(
        outSize * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::MatrixMultiply(
       device_,
       betann::GetDataType<T>(),
       out,
       device_.CreateBufferFromVector(a),
       aShape,
       aStrides,
       device_.CreateBufferFromVector(b),
       bShape,
       bStrides);
    device_.Flush();
    return ReadFromBuffer<T>(out, outSize);
  }

};

TEST_F(MatrixMultiplyTest, GEMVContiguous) {
  auto a = RandomNumbers<float>(20, 10);
  auto b = RandomNumbers<float>(4, 10);
  // matmul(a, b)
  EXPECT_EQ(GpuMatmul(a, {5, 4}, {4, 1},
                      b, {4, 1}, {1, 1}),
            CpuMatmul(a, {5, 4}, {4, 1},
                      b, {4, 1}, {1, 1}));
  // matmul(a.T, b)
  EXPECT_EQ(GpuMatmul(a, {5, 4}, {1, 5},
                      b, {4, 1}, {1, 4}),
            CpuMatmul(a, {5, 4}, {1, 5},
                      b, {4, 1}, {1, 4}));
  // matmul(b, a)
  EXPECT_EQ(GpuMatmul(b, {1, 4}, {4, 1},
                      a, {4, 5}, {5, 1}),
            CpuMatmul(b, {1, 4}, {4, 1},
                      a, {4, 5}, {5, 1}));
  // matmul(b, a.T)
  EXPECT_EQ(GpuMatmul(b, {1, 4}, {4, 1},
                      a, {4, 5}, {1, 4}),
            CpuMatmul(b, {1, 4}, {4, 1},
                      a, {4, 5}, {1, 4}));
}

TEST_F(MatrixMultiplyTest, GEMVContiguousBatch) {
  auto a = RandomNumbers<float>(10 * 20, 10);
  auto b = RandomNumbers<float>(10 * 4, 10);
  // matmul(10, a, b)
  EXPECT_EQ(GpuMatmul(a, {10, 5, 4}, {20, 4, 1},
                      b, {10, 4, 1}, {4, 1, 1}),
            CpuMatmul(a, {10, 5, 4}, {20, 4, 1},
                      b, {10, 4, 1}, {4, 1, 1}));
  // matmul(10, a.T, b)
  EXPECT_EQ(GpuMatmul(a, {10, 5, 4}, {20, 1, 5},
                      b, {10, 4, 1}, {4, 1, 1}),
            CpuMatmul(a, {10, 5, 4}, {20, 1, 5},
                      b, {10, 4, 1}, {4, 1, 1}));
  // matmul(10, b, a)
  EXPECT_EQ(GpuMatmul(b, {10, 1, 4}, {4, 1, 1},
                      a, {10, 4, 5}, {20, 5, 1}),
            CpuMatmul(b, {10, 1, 4}, {4, 1, 1},
                      a, {10, 4, 5}, {20, 5, 1}));
  // matmul(10, b, a.T)
  EXPECT_EQ(GpuMatmul(b, {10, 1, 4}, {4, 1, 1},
                      a, {10, 4, 5}, {20, 1, 4}),
            CpuMatmul(b, {10, 1, 4}, {4, 1, 1},
                      a, {10, 4, 5}, {20, 1, 4}));
  // matmul(2, 5, a, b)
  EXPECT_EQ(GpuMatmul(a, {2, 5, 5, 4}, {100, 20, 4, 1},
                      b, {2, 5, 4, 1}, {20, 4, 1, 1}),
            CpuMatmul(a, {2, 5, 5, 4}, {100, 20, 4, 1},
                      b, {2, 5, 4, 1}, {20, 4, 1, 1}));
  // matmul(2, 5, a.T, b)
  EXPECT_EQ(GpuMatmul(a, {2, 5, 5, 4}, {100, 20, 1, 5},
                      b, {2, 5, 4, 1}, {20, 4, 1, 1}),
            CpuMatmul(a, {2, 5, 5, 4}, {100, 20, 1, 5},
                      b, {2, 5, 4, 1}, {20, 4, 1, 1}));
  // matmul(2, 5, b, a)
  EXPECT_EQ(GpuMatmul(b, {2, 5, 1, 4}, {20, 4, 1, 1},
                      a, {2, 5, 4, 5}, {100, 20, 5, 1}),
            CpuMatmul(b, {2, 5, 1, 4}, {20, 4, 1, 1},
                      a, {2, 5, 4, 5}, {100, 20, 5, 1}));
}

TEST_F(MatrixMultiplyTest, GEMVNonContiguous) {
  // matmul(a_broadcast, b)
  auto a = RandomNumbers<float>(4, 10);
  auto b = RandomNumbers<float>(4, 10);
  EXPECT_EQ(GpuMatmul(a, {5, 4}, {0, 1},
                      b, {4, 1}, {1, 1}),
            CpuMatmul(a, {5, 4}, {0, 1},
                      b, {4, 1}, {1, 1}));
  // matmul(a_broadcast.T, b)
  b = RandomNumbers<float>(5, 10);
  EXPECT_EQ(GpuMatmul(a, {4, 5}, {1, 0},
                      b, {5, 1}, {1, 1}),
            CpuMatmul(a, {4, 5}, {1, 0},
                      b, {5, 1}, {1, 1}));
  // matmul(a_broadcast, b_broadcast)
  b = RandomNumbers<float>(1, 10);
  EXPECT_EQ(GpuMatmul(a, {5, 4}, {0, 1},
                      b, {4, 1}, {0, 0}),
            CpuMatmul(a, {5, 4}, {0, 1},
                      b, {4, 1}, {0, 0}));
  // matmul(a_broadcast.T, b_broadcast)
  EXPECT_EQ(GpuMatmul(a, {4, 5}, {1, 0},
                      b, {5, 1}, {0, 0}),
            CpuMatmul(a, {4, 5}, {1, 0},
                      b, {5, 1}, {0, 0}));
  // matmul(a, b_broadcast)
  a = RandomNumbers<float>(20, 10);
  EXPECT_EQ(GpuMatmul(a, {5, 4}, {4, 1},
                      b, {4, 1}, {0, 0}),
            CpuMatmul(a, {5, 4}, {4, 1},
                      b, {4, 1}, {0, 0}));
  // matmul(a.T, b_broadcast)
  EXPECT_EQ(GpuMatmul(a, {4, 5}, {1, 4},
                      b, {5, 1}, {0, 0}),
            CpuMatmul(a, {4, 5}, {1, 4},
                      b, {5, 1}, {0, 0}));
}

TEST_F(MatrixMultiplyTest, GEMVNonContiguousBatch) {
  // matmul(10, a, b)
  auto a = RandomNumbers<float>(20, 10);
  auto b = RandomNumbers<float>(4, 10);
  EXPECT_EQ(GpuMatmul(a, {10, 5, 4}, {0, 4, 1},
                      b, {10, 4, 1}, {0, 1, 1}),
            CpuMatmul(a, {10, 5, 4}, {0, 4, 1},
                      b, {10, 4, 1}, {0, 1, 1}));
  // matmul(2, 5, a.T, b)
  a = RandomNumbers<float>(5 * 20, 10);
  b = RandomNumbers<float>(5 * 4, 10);
  EXPECT_EQ(GpuMatmul(a, {2, 5, 5, 4}, {0, 20, 1, 5},
                      b, {2, 5, 4, 1}, {0, 4, 1, 1}),
            CpuMatmul(a, {2, 5, 5, 4}, {0, 20, 1, 5},
                      b, {2, 5, 4, 1}, {0, 4, 1, 1}));
  // matmul(2, 5, b, a)
  EXPECT_EQ(GpuMatmul(b, {2, 5, 1, 4}, {0, 4, 1, 1},
                      a, {2, 5, 4, 5}, {0, 20, 5, 1}),
            CpuMatmul(b, {2, 5, 1, 4}, {0, 4, 1, 1},
                      a, {2, 5, 4, 5}, {0, 20, 5, 1}));
  // matmul(2, 5, b, a.T)
  EXPECT_EQ(GpuMatmul(b, {2, 5, 1, 4}, {0, 4, 1, 1},
                      a, {2, 5, 4, 5}, {0, 20, 1, 4}),
            CpuMatmul(b, {2, 5, 1, 4}, {0, 4, 1, 1},
                      a, {2, 5, 4, 5}, {0, 20, 1, 4}));
}
