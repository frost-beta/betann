#include "betann_tests.h"

#include <fmt/format.h>

class MatrixVectorMultiplyTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> GpuGemv(const std::vector<T>& mat,
                         const std::vector<uint32_t>& shape,
                         const std::vector<T>& vec) {
    uint32_t outSize = betann::NumElements(shape) / shape[shape.size() - 1];
    wgpu::Buffer out = device_.CreateBuffer(
        outSize * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::MatrixVectorMultiply(device_,
                                 betann::GetDataType<T>(),
                                 out,
                                 device_.CreateBufferFromVector(mat),
                                 shape,
                                 device_.CreateBufferFromVector(vec));
    device_.Flush();
    return ReadFromBuffer<T>(out, outSize);
  }

  template<typename T>
  std::vector<T> CpuGemv(const std::vector<T>& mat,
                         uint32_t matRows,
                         uint32_t matCols,
                         const std::vector<T>& vec) {
    std::vector<T> result(matRows, 0);
    for (uint32_t i = 0; i < matRows; i++) {
      for (uint32_t j = 0; j < matCols; j++) {
        result[i] += mat[i * matCols + j] * vec[j];
      }
    }
    return result;
  }
};

TEST_F(MatrixVectorMultiplyTests, Contiguous) {
  const uint32_t shapes[][2] = {
    {1, 1},
    {4, 1},
    {4, 2},
    {5, 1},
    {5, 5},
    {3, 31},
    {4, 32},
    {5, 33},
    {17, 129},
    {16 * 2 + 1, 32 * 8 + 1},
    {100, 100},
  };
  for (auto [M, N] : shapes) {
    auto a = RandomNumbers<int32_t>(M * N);
    auto b = RandomNumbers<int32_t>(N);
    SCOPED_TRACE(fmt::format("Shape: {}x{}", M, N));
    EXPECT_EQ(GpuGemv(a, {M, N}, b), CpuGemv(a, M, N, b));
  }
}

TEST_F(MatrixVectorMultiplyTests, ContiguousBatches) {
  const uint32_t shapes[][3] = {
    {2, 2, 2},
    {2, 5, 33},
    {2, 16, 128},
    {2, 17, 129},
    {8, 16 * 2 + 7, 32 * 8 + 7},
  };
  for (auto [B, M, N] : shapes) {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<std::vector<float>> batchX;
    std::vector<std::vector<float>> batchY;
    for (uint32_t b = 0; b < B; ++b) {
      batchX.push_back(RandomNumbers<float>(M * N, 10));
      x.insert(x.end(), batchX.back().begin(), batchX.back().end());
      batchY.push_back(RandomNumbers<float>(N, 10));
      y.insert(y.end(), batchY.back().begin(), batchY.back().end());
    }
    std::vector<float> z;
    for (uint32_t b = 0; b < B; ++b) {
      auto result = CpuGemv(batchX[b], M, N, batchY[b]);
      z.insert(z.end(), result.begin(), result.end());
    }
    SCOPED_TRACE(fmt::format("Batch: {}, Shape: {}x{}", B, M, N));
    EXPECT_EQ(GpuGemv(x, {B, M ,N}, y), z);
  }
}
