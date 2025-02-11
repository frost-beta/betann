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
    wgpu::Buffer out = device_.CreateBuffer(
        outSize * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
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

TEST_F(MatrixMultiplyTest, GEMV) {
  auto a = RandomNumbers<float>(16, 10);
  auto b = RandomNumbers<float>(4, 10);
  // gemv(a, b)
  EXPECT_EQ(GpuMatmul(a, {4, 4}, {4, 1},
                      b, {4, 1}, {1, 0}),
            CpuMatmul(a, {4, 4}, {4, 1},
                      b, {4, 1}, {1, 0}));
  // gemv(b, a)
  EXPECT_EQ(GpuMatmul(b, {1, 4}, {0, 1},
                      a, {4, 4}, {4, 1}),
            CpuMatmul(b, {1, 4}, {0, 1},
                      a, {4, 4}, {4, 1}));
  // gemv(a.T, b)
  EXPECT_EQ(GpuMatmul(a, {4, 4}, {1, 4},
                      b, {4, 1}, {1, 0}),
            CpuMatmul(a, {4, 4}, {1, 4},
                      b, {4, 1}, {1, 0}));
  // gemv(b, a.T)
  EXPECT_EQ(GpuMatmul(b, {1, 4}, {0, 1},
                      a, {4, 4}, {1, 4}),
            CpuMatmul(b, {1, 4}, {0, 1},
                      a, {4, 4}, {1, 4}));
}

TEST_F(MatrixMultiplyTest, BatchGEMV) {
  auto a = RandomNumbers<float>(10 * 16, 10);
  auto b = RandomNumbers<float>(10 * 4, 10);
  // gemv(10, a, b)
  EXPECT_EQ(GpuMatmul(a, {10, 4, 4}, {16, 4, 1},
                      b, {10, 4, 1}, {4, 1, 0}),
            CpuMatmul(a, {10, 4, 4}, {16, 4, 1},
                      b, {10, 4, 1}, {4, 1, 0}));
  // gemv(2, 5, a, b)
  EXPECT_EQ(GpuMatmul(a, {2, 5, 4, 4}, {80, 16, 4, 1},
                      b, {2, 5, 4, 1}, {20, 4, 1, 0}),
            CpuMatmul(a, {2, 5, 4, 4}, {80, 16, 4, 1},
                      b, {2, 5, 4, 1}, {20, 4, 1, 0}));
}
