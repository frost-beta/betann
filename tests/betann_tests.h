#include <cstring>
#include <random>

#include <betann/betann.h>
#include <gtest/gtest.h>

class BetaNNTests : public testing::Test {
 protected:
  template<typename T>
  std::vector<T> ReadFromBuffer(const wgpu::Buffer& buf, size_t size) {
    wgpu::Buffer staging = device_.CopyToStagingBuffer(buf);
    device_.Flush();
    std::vector<T> out(size);
    device_.WaitFor(device_.ReadStagingBuffer(
        staging,
        [&](const void* data) {
          std::memcpy(out.data(), data, size * sizeof(T));
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
};
