#include <algorithm>
#include <cstring>
#include <numeric>

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

  betann::Device device_;
};
