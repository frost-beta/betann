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
                      device_.CreateBufferFromVector(input),
                      input.size(),
                      disableSubgroups);
    device_.Flush();
    return ReadFromBuffer<T>(output, 1)[0];
  }

  template<typename T, typename U>
  std::vector<T> RunReduceRow(betann::ReduceType type,
                              const std::vector<U>& input,
                              const std::vector<uint32_t>& shape,
                              const std::vector<uint32_t>& strides,
                              const std::vector<uint32_t>& axes,
                              bool disableSubgroups = false) {
    uint32_t outputNumElements =
        betann::NumElements(betann::RemoveIndices(shape, axes));
    betann::Buffer output = device_.CreateBuffer(
        outputNumElements * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReduceRow(device_,
                      type,
                      betann::GetDataType<T>(),
                      output,
                      outputNumElements,
                      betann::GetDataType<U>(),
                      device_.CreateBufferFromVector(input),
                      shape,
                      strides,
                      axes,
                      betann::KeepIndices(shape, axes),
                      betann::KeepIndices(strides, axes),
                      disableSubgroups);
    device_.Flush();
    return ReadFromBuffer<T>(output, outputNumElements);
  }

  std::vector<uint32_t> Strides(const std::vector<uint32_t>& shape) {
    std::vector<uint32_t> strides(shape.size());
    uint32_t size = 1;
    for (int32_t i = shape.size() - 1; i >= 0; --i) {
      strides[i] = size;
      size *= shape[i];
    }
    return strides;
  }

  template<typename T>
  std::vector<T> Sum(const std::vector<T>& input,
                     const std::vector<uint32_t>& shape,
                     const std::vector<uint32_t>& strides,
                     const std::vector<uint32_t>& axes) {
    std::vector<T> output(
        betann::NumElements(betann::RemoveIndices(shape, axes)), 0);
    for (size_t i = 0; i < input.size(); ++i) {
      uint32_t outIndex = 0;
      uint32_t inputIndex = i;
      for (int32_t axis = shape.size() - 1, multiplier = 1; axis >= 0; --axis) {
        if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
          if (strides[axis] != 0) {
            outIndex += (inputIndex / strides[axis]) * multiplier;
            inputIndex %= strides[axis];
          }
          multiplier *= shape[axis];
        }
      }
      output[outIndex] += input[i];
    }
    return output;
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

TEST_F(ReduceTests, ReduceRow) {
  const bool disableSubgroups = true;
  const std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> shapes[] = {
    {{31, 127}, {1}},
    {{31, 127}, {0, 1}},
    {{32, 128}, {1}},
    {{32, 128}, {0, 1}},
    {{33, 129}, {1}},
    {{33, 129}, {0, 1}},
    {{31, 127, 127}, {2}},
    {{31, 127, 127}, {1, 2}},
    {{33, 33, 33, 1}, {3}},
    {{33, 33, 33, 1}, {2, 3}},
    {{33, 33, 33, 1}, {1, 2, 3}},
  };
  for (const auto& [shape, axes] : shapes) {
    SCOPED_TRACE(fmt::format("Subgroups: {}, shape: {}, axes: {}",
                             !disableSubgroups,
                             VecToString(shape),
                             VecToString(axes)));
    auto strides = Strides(shape);
    auto ints = RandomNumbers<int32_t>(betann::NumElements(shape), 10);
    EXPECT_EQ(RunReduceRow<int32_t>(betann::ReduceType::Sum,
                                    ints, shape, strides, axes,
                                    disableSubgroups),
              Sum(ints, shape, strides, axes));
  }
}
