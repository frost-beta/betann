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
  std::vector<T> RandomNumbers(size_t size) {
    std::uniform_int_distribution<T> dist(0, 8964);
    std::vector<T> ret;
    for (size_t i = 0; i < size; ++i)
      ret.push_back(dist(mt_));
    return ret;
  }

  betann::Device device_;

 private:
  std::mt19937 mt_;
};
