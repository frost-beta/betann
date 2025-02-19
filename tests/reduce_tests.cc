#include "betann_tests.h"

#include <fmt/format.h>

#include "betann/reduce.h"

class ReduceTests : public BetaNNTests {
 public:
  template<typename T, typename U>
  T RunReduceAll(betann::ReduceType type,
                 const std::vector<U>& input,
                 bool disableSubgroups = false) {
    betann::Buffer output = device_.CreateBuffer(
        sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReduceAll(device_,
                      type,
                      betann::GetDataType<T>(),
                      output,
                      betann::GetDataType<U>(),
                      device_.CreateBufferTransformTo<U>(input),
                      input.size(),
                      disableSubgroups);
    device_.Flush();
    return ReadFromBuffer<T>(output, 1)[0];
  }

  std::vector<bool> GetParameters() {
    std::vector<bool> disableSubgroups{true};
    if (device_.SupportsSubgroups())
      disableSubgroups.push_back(false);
    return disableSubgroups;
  }
};

TEST_F(ReduceTests, ReduceAll) {
  const uint32_t sizes[] = {1, 31, 32, 33, 127, 128, 129, 200, 300, 4100};
  for (bool disableSubgroups : GetParameters()) {
    for (uint32_t size : sizes) {
      SCOPED_TRACE(fmt::format("Subgroups: {}, size: {}",
                               !disableSubgroups, size));
      auto floats = RandomNumbers<float>(size);
      EXPECT_EQ(RunReduceAll<float>(betann::ReduceType::Min, floats,
                                    disableSubgroups),
                *std::min_element(floats.begin(), floats.end()));
      EXPECT_EQ(RunReduceAll<float>(betann::ReduceType::Max, floats,
                                    disableSubgroups),
                *std::max_element(floats.begin(), floats.end()));
      auto ints = RandomNumbers<int32_t>(size, 10);
      EXPECT_EQ(RunReduceAll<int32_t>(betann::ReduceType::Sum, ints,
                                      disableSubgroups),
                std::accumulate(ints.begin(), ints.end(), 0));
      EXPECT_EQ(RunReduceAll<uint32_t>(betann::ReduceType::Prod, ints,
                                       disableSubgroups),
                std::accumulate(ints.begin(), ints.end(), 1,
                                std::multiplies<int32_t>()));
    }
    EXPECT_EQ(RunReduceAll<uint32_t>(betann::ReduceType::And,
                                     std::vector<uint32_t>{true, false},
                                     disableSubgroups),
              false);
    EXPECT_EQ(RunReduceAll<uint32_t>(betann::ReduceType::Or,
                                     std::vector<uint32_t>{true, false},
                                     disableSubgroups),
              true);
  }
}
