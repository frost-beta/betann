#include "betann_tests.h"

class MatrixMultiplyTest : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> MatmulGpu(const std::vector<T>& a,
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

  template<typename T>
  std::vector<T> MatmulCpu(const std::vector<T>& a,
                           const std::vector<uint32_t>& aShape,
                           const std::vector<uint32_t>& aStrides,
                           const std::vector<T>& b,
                           const std::vector<uint32_t>& bShape,
                           const std::vector<uint32_t>& bStrides) {
    uint32_t batchSize = 1;
    for (size_t i = 0; i < aShape.size() - 2; ++i) {
      batchSize *= aShape[i];
    }
    uint32_t m = aShape[aShape.size() - 2];
    uint32_t n = bShape[bShape.size() - 1];
    uint32_t k = aShape[aShape.size() - 1];
    std::vector<T> result(batchSize * m * n, 0);
    for (uint32_t batch = 0; batch < batchSize; ++batch) {
      for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
          T temp = 0;
          for (uint32_t l = 0; l < k; ++l) {
            uint32_t aIndex = 0;
            uint32_t bIndex = 0;
            uint32_t batchStride = 1;
            for (size_t dim = 0; dim < aShape.size() - 2; ++dim) {
              uint32_t batchIndex = (batch / batchStride) % aShape[dim];
              aIndex += batchIndex * aStrides[dim];
              bIndex += batchIndex * bStrides[dim];
              batchStride *= aShape[dim];
            }
            aIndex += i * aStrides[aShape.size() - 2] +
                      l * aStrides[aShape.size() - 1];
            bIndex += l * bStrides[bShape.size() - 2] +
                      j * bStrides[bShape.size() - 1];
            temp += a[aIndex] * b[bIndex];
          }
          uint32_t resultIndex = batch * m * n + i * n + j;
          result[resultIndex] = temp;
        }
      }
    }
    return result;
  }
};

TEST_F(MatrixMultiplyTest, GEMV) {
  auto a = RandomNumbers<float>(16, 10);
  auto b = RandomNumbers<float>(4, 10);
  // gemv(a, b)
  EXPECT_EQ(MatmulGpu(a, {4, 4}, {4, 1},
                      b, {4, 1}, {1, 0}),
            MatmulCpu(a, {4, 4}, {4, 1},
                      b, {4, 1}, {1, 0}));
  // gemv(b, a)
  EXPECT_EQ(MatmulGpu(b, {1, 4}, {0, 1},
                      a, {4, 4}, {4, 1}),
            MatmulCpu(b, {1, 4}, {0, 1},
                      a, {4, 4}, {4, 1}));
  // gemv(a.T, b)
  EXPECT_EQ(MatmulGpu(a, {4, 4}, {1, 4},
                      b, {4, 1}, {1, 0}),
            MatmulCpu(a, {4, 4}, {1, 4},
                      b, {4, 1}, {1, 0}));
  // gemv(b, a.T)
  EXPECT_EQ(MatmulGpu(b, {1, 4}, {0, 1},
                      a, {4, 4}, {1, 4}),
            MatmulCpu(b, {1, 4}, {0, 1},
                      a, {4, 4}, {1, 4}));
}
