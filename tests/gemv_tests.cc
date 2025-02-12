#include "betann_tests.h"

#include <fmt/format.h>

#include "betann/matmul.h"

class MatrixVectorMultiplyTests : public BetaNNTests {
 public:
  template<typename T>
  std::vector<T> GpuGemv(const std::vector<T>& mat,
                         const std::vector<uint32_t>& shape,
                         const std::vector<uint32_t>& matStrides,
                         const std::vector<T>& vec,
                         const std::vector<uint32_t>& vecStrides,
                         bool disableSubgroups = false,
                         bool matTranspose = false) {
    uint32_t vecSize = shape[shape.size() - 1];
    uint32_t outSize = betann::NumElements(shape) / vecSize;
    wgpu::Buffer out = device_.CreateBuffer(
        outSize * sizeof(T),
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    betann::MatrixVectorMultiply(
        device_,
       betann::GetDataType<T>(),
       betann::Slice(shape, 0, -2),
       out,
       device_.CreateBufferFromVector(mat),
       matTranspose,
       matTranspose ? shape[shape.size() - 1]
                    : shape[shape.size() - 2],
       matTranspose ? shape[shape.size() - 2]
                    : shape[shape.size() - 1],
       matTranspose ? matStrides[matStrides.size() - 1]
                    : matStrides[matStrides.size() - 2],
       betann::Slice(matStrides, 0, -2),
       device_.CreateBufferFromVector(vec),
       betann::Slice(vecStrides, 0, -2),
       disableSubgroups);
    device_.Flush();
    return ReadFromBuffer<T>(out, outSize);
  }

  template<typename T>
  std::vector<T> GpuGemvt(const std::vector<T>& mat,
                          const std::vector<uint32_t>& shape,
                          const std::vector<uint32_t>& matStrides,
                          const std::vector<T>& vec,
                          const std::vector<uint32_t>& vecStrides,
                          bool disableSubgroups = false) {
    return GpuGemv(mat, shape, matStrides, vec, vecStrides, disableSubgroups,
                   true);
  }

  std::vector<bool> GetParameters() {
    std::vector<bool> disableSubgroups{true};
    if (device_.SupportsSubgroups())
      disableSubgroups.push_back(false);
    return disableSubgroups;
  }

  std::vector<bool> GetTransposeParameters() {
    std::vector<bool> disableSubgroups{true};
#ifdef __APPLE__
    if (device_.SupportsSubgroups())
      disableSubgroups.push_back(false);
#endif
    return disableSubgroups;
  }
};

TEST_F(MatrixVectorMultiplyTests, Contiguous) {
  for (bool disableSubgroups : GetParameters()) {
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
      {5000, 100},
    };
    for (auto [M, K] : shapes) {
      auto a = RandomNumbers<int32_t>(M * K, 10);
      auto b = RandomNumbers<int32_t>(K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Shape: {}x{}",
                               !disableSubgroups, M, K));
      EXPECT_EQ(GpuGemv(a, {M, K}, {K, 1}, b, {1, 0}, disableSubgroups),
                CpuMatmul(a, {M, K}, {K, 1}, b, {K, 1}, {1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, NonContiguous) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][2] = {
      {4, 2},
      {17, 33},
      {33, 17},
      {33, 129},
      {129, 200},
    };
    for (auto [M, K] : shapes) {
      auto a = RandomNumbers<int32_t>(K, 10);
      auto b = RandomNumbers<int32_t>(K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Shape: {}x{}",
                               !disableSubgroups, M, K));
      EXPECT_EQ(GpuGemv(a, {M, K}, {0, 1}, b, {1, 1}, disableSubgroups),
                CpuMatmul(a, {M, K}, {0, 1}, b, {K, 1}, {1, 1}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, ContiguousBatches) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][3] = {
      {2, 2, 2},
      {2, 5, 33},
      {2, 16, 128},
      {2, 17, 129},
      {8, 16 * 2 + 7, 32 * 8 + 7},
    };
    for (auto [B, M, K] : shapes) {
      auto x = RandomNumbers<float>(B * M * K, 10);
      auto y = RandomNumbers<float>(B * K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Batch: {}, Shape: {}x{}",
                               !disableSubgroups, B, M, K));
      EXPECT_EQ(GpuGemv(x, {B, M ,K}, {M * K, K, 1},
                        y, {K, 1, 10},
                        disableSubgroups),
                CpuMatmul(x, {B, M, K}, {M * K, K, 1},
                          y, {B, K, 1}, {K, 1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, NonContiguousBatches) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][4] = {
      {1, 2, 2, 1},
      {2, 2, 4, 4},
      {7, 9, 33, 129},
    };
    for (auto [A, B, M, K] : shapes) {
      auto x = RandomNumbers<float>(A * B * M * K, 10);
      auto y = RandomNumbers<float>(A * B * K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Batch: {}x{}, Shape: {}x{}",
                               !disableSubgroups, A, B, M, K));
      EXPECT_EQ(GpuGemv(x, {A, B, M ,K}, {B * M * K, M * K, K, 1},
                        y, {B * K, K, 1, 0},
                        disableSubgroups),
                CpuMatmul(x, {A, B, M, K}, {B * M * K, M * K, K, 1},
                          y, {A, B, K, 1}, {B * K, K, 1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, VirtualBatches) {
  for (bool disableSubgroups : GetParameters()) {
    uint32_t M = 5, K = 4;
    auto x = RandomNumbers<float>(M * K, 10);
    auto y = RandomNumbers<float>(K, 10);
    EXPECT_EQ(GpuGemv(x, {10, M, K}, {0, K, 1},
                      y, {0, 1, 1},
                      disableSubgroups),
              CpuMatmul(x, {10, M, K}, {0, K, 1},
                        y, {10, K, 1}, {0, 1, 1}));
    uint32_t B = 2;
    x = RandomNumbers<float>(B * M * K, 10);
    y = RandomNumbers<float>(B * K, 10);
    EXPECT_EQ(GpuGemv(x, {8, B, M, K}, {0, M * K, K, 1},
                      y, {0, K, 1, 1},
                      disableSubgroups),
              CpuMatmul(x, {8, B, M, K}, {0, M * K, K, 1},
                        y, {8, B, K, 1}, {0, K, 1, 1}));
  }
}

TEST_F(MatrixVectorMultiplyTests, TransposeContiguous) {
  for (bool disableSubgroups : GetTransposeParameters()) {
    const uint32_t shapes[][2] = {
      {1, 1},
      {1, 4},
      {1, 9},
      {2, 4},
      {2, 9},
      {5, 9},
      {31, 7},
      {31, 8},
      {31, 9},
      {129, 17},
      {32 * 8 + 1, 16 * 2 + 1},
      {100, 100},
      {100, 600},
      {100, 2100},
    };
    for (auto [M, K] : shapes) {
      auto x = RandomNumbers<int32_t>(M * K, 10);
      auto y = RandomNumbers<int32_t>(M, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Shape: {}x{}",
                               !disableSubgroups, M, K));
      EXPECT_EQ(GpuGemvt(x, {K, M}, {1, K}, y, {1, 0}, disableSubgroups),
                CpuMatmul(x, {K, M}, {1, K}, y, {M, 1}, {1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, TranposeNonContiguous) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][2] = {
      {4, 2},
      {17, 33},
      {33, 17},
      {33, 129},
      {129, 200},
    };
    for (auto [M, K] : shapes) {
      auto a = RandomNumbers<int32_t>(M, 10);
      auto b = RandomNumbers<int32_t>(K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Shape: {}x{}",
                               !disableSubgroups, M, K));
      EXPECT_EQ(GpuGemvt(a, {M, K}, {1, 0}, b, {1, 1}, disableSubgroups),
                CpuMatmul(a, {M, K}, {1, 0}, b, {K, 1}, {1, 1}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, TransposeContiguousBatches) {
  for (bool disableSubgroups : GetTransposeParameters()) {
    const uint32_t shapes[][3] = {
      {2, 2, 1},
      {2, 33, 5},
      {2, 128, 16},
      {2, 129, 17},
      {8, 32 * 8 + 7, 16 * 2 + 7},
    };
    for (auto [B, M, K] : shapes) {
      auto x = RandomNumbers<float>(B * M * K, 10);
      auto y = RandomNumbers<float>(B * M, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Batch: {}, Shape: {}x{}",
                               !disableSubgroups, B, M, K));
      EXPECT_EQ(GpuGemvt(x, {B, K, M}, {M * K, 1, K},
                         y, {M, 1, 0},
                         disableSubgroups),
                CpuMatmul(x, {B, K, M}, {M * K, 1, K},
                          y, {B, M, 1}, {M, 1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, TranposeNonContiguousBatches) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][4] = {
      {1, 2, 2, 1},
      {2, 2, 4, 4},
      {7, 9, 129, 33},
    };
    for (auto [A, B, M, K] : shapes) {
      auto x = RandomNumbers<float>(A * B * M * K, 10);
      auto y = RandomNumbers<float>(A * B * M, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Batch: {}x{}, Shape: {}x{}",
                               !disableSubgroups, A, B, M, K));
      EXPECT_EQ(GpuGemvt(x, {A, B, K ,M}, {B * M * K, M * K, 1, K},
                         y, {B * M, M, 1, 0},
                         disableSubgroups),
                CpuMatmul(x, {A, B, K, M}, {B * M * K, M * K, 1, K},
                          y, {A, B, M, 1}, {B * M, M, 1, 0}));
    }
  }
}

TEST_F(MatrixVectorMultiplyTests, TransposeVirtualBatches) {
  for (bool disableSubgroups : GetParameters()) {
    uint32_t M = 5, K = 4;
    auto x = RandomNumbers<float>(M * K, 10);
    auto y = RandomNumbers<float>(M, 10);
    EXPECT_EQ(GpuGemvt(x, {10, K, M}, {0, 1, K},
                       y, {0, 1, 1},
                       disableSubgroups),
              CpuMatmul(x, {10, K, M}, {0, 1, K},
                        y, {10, M, 1}, {0, 1, 1}));
    uint32_t B = 2;
    x = RandomNumbers<float>(B * M * K, 10);
    y = RandomNumbers<float>(B * M, 10);
    EXPECT_EQ(GpuGemvt(x, {8, B, K, M}, {0, M * K, 1, K},
                       y, {0, M, 1, 1},
                       disableSubgroups),
              CpuMatmul(x, {8, B, K, M}, {0, M * K, 1, K},
                        y, {8, B, M, 1}, {0, M, 1, 1}));
  }
}
