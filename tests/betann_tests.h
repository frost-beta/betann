#include <cstring>
#include <random>

#include <betann/betann.h>
#include <gtest/gtest.h>

class BetaNNTests : public testing::Test {
 protected:
  template<typename T>
  std::vector<T> ReadFromBuffer(const betann::Buffer& buf, size_t size) {
    betann::Buffer staging = device_.CopyToStagingBuffer(buf);
    device_.Flush();
    std::vector<T> out(size);
    device_.WaitFor(device_.ReadFullStagingBuffer(
        staging,
        [&](const void* data, uint64_t size, uint64_t offset) {
          std::memcpy(out.data(),
                      static_cast<const char*>(data) + offset,
                      size);
        }));
    return out;
  }

  template<typename T>
  std::vector<T> RandomNumbers(size_t size, int upper = 8964) {
    std::uniform_int_distribution<int32_t> dist(2, upper);
    std::vector<T> ret;
    for (size_t i = 0; i < size; ++i)
      ret.push_back(static_cast<T>(dist(mt_)));
    return ret;
  }

  template<typename T>
  std::vector<T> CpuMatmul(const std::vector<T>& a,
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
            for (int batchIndex = batch, dim = aShape.size() - 3;
                 dim >= 0 && batchIndex > 0;
                 dim--) {
              uint32_t dimIndex = batchIndex % aShape[dim];
              aIndex += dimIndex * aStrides[dim];
              bIndex += dimIndex * bStrides[dim];
              batchIndex /= aShape[dim];
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

  template<typename T>
  std::vector<T> Iota(size_t size, T start) {
    std::vector<T> a(size);
    std::iota(a.begin(), a.end(), start);
    return a;
  }

  template<typename T, typename F>
  std::vector<T> Map(std::vector<T> vec, F&& transform) {
    std::transform(vec.begin(), vec.end(), vec.begin(), transform);
    return vec;
  }

  template<typename T, typename... Args>
  std::vector<T> Concat(std::vector<T> a, Args&&... args) {
    ((a.insert(a.end(), args.begin(), args.end())), ...);
    return a;
  }

  betann::Device device_;

 private:
  std::mt19937 mt_;

  // Disable the CollapseContiguousDims optimization to accurately test kernels.
  betann::DisableCollapseDims disable_collapse_dims_;
};
