#include "betann_tests.h"

#include "betann/reduce.h"

class ReduceTests : public BetaNNTests {
 public:
  template<typename T, typename U>
  T RunReduceAll(const char* op, const std::vector<U>& input) {
    betann::Buffer output = device_.CreateBuffer(
        sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReduceAll(device_,
                      op,
                      betann::GetDataType<T>(),
                      output,
                      betann::GetDataType<U>(),
                      device_.CreateBufferTransformTo<U>(input),
                      input.size());
    device_.Flush();
    return ReadFromBuffer<T>(output, 1)[0];
  }
};

TEST_F(ReduceTests, ReduceAll) {
  const uint32_t sizes[] = {1, 31, 32, 33, 127, 128, 129, 200};
  for (uint32_t size : sizes) {
    auto floats = RandomNumbers<float>(size);
    EXPECT_EQ(RunReduceAll<float>("sum", floats),
              std::accumulate(floats.begin(), floats.end(), 0));
    EXPECT_EQ(RunReduceAll<float>("min", floats),
              *std::min_element(floats.begin(), floats.end()));
    EXPECT_EQ(RunReduceAll<float>("max", floats),
              *std::max_element(floats.begin(), floats.end()));
    auto ints = RandomNumbers<int32_t>(size, 10);
    EXPECT_EQ(RunReduceAll<uint32_t>("product", ints),
              std::accumulate(ints.begin(), ints.end(), 1,
                              std::multiplies<int32_t>()));
  }
  EXPECT_EQ(RunReduceAll<uint32_t>("and", std::vector<uint32_t>{true, false}),
            false);
  EXPECT_EQ(RunReduceAll<uint32_t>("or", std::vector<uint32_t>{true, false}),
            true);
}
