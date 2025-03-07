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
  std::vector<T> RunReduceLast(betann::ReduceType type,
                               const std::vector<U>& input,
                               const std::vector<uint32_t>& shape,
                               const std::vector<uint32_t>& axes,
                               bool disableSubgroups = false) {
    uint32_t outputNumElements =
        betann::NumElements(betann::RemoveIndices(shape, axes));
    uint32_t rowSize =
        betann::NumElements(betann::KeepIndices(shape, axes));
    betann::Buffer output = device_.CreateBuffer(
        outputNumElements * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReduceLast(device_,
                       type,
                       betann::GetDataType<T>(),
                       output,
                       outputNumElements,
                       betann::GetDataType<U>(),
                       device_.CreateBufferFromVector(input),
                       rowSize,
                       disableSubgroups);
    device_.Flush();
    return ReadFromBuffer<T>(output, outputNumElements);
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

  template<typename T>
  std::vector<T> RunReduceNone(betann::ReduceType type,
                               uint32_t outputNumElements) {
    betann::Buffer output = device_.CreateBuffer(
        betann::DivCeil(outputNumElements, 4u) * 4 * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReduceNone(device_,
                       type,
                       betann::GetDataType<T>(),
                       output,
                       outputNumElements);
    device_.Flush();
    return ReadFromBuffer<T>(output, outputNumElements);
  }

  template<typename T, typename U>
  std::vector<T> RunReduce(betann::ReductionPlanType planType,
                           betann::ReduceType type,
                           const std::vector<U>& input,
                           const std::vector<uint32_t>& shape,
                           const std::vector<uint32_t>& strides,
                           const std::vector<uint32_t>& axes) {
    uint32_t outputNumElements =
        betann::NumElements(betann::RemoveIndices(shape, axes));
    betann::Buffer output = device_.CreateBuffer(
        outputNumElements * sizeof(T),
        betann::BufferUsage::Storage | betann::BufferUsage::CopySrc);
    betann::ReductionPlan plan = {
      planType,
      betann::KeepIndices(shape, axes),
      betann::KeepIndices(strides, axes),
    };
    betann::Reduce(device_,
                   std::move(plan),
                   type,
                   betann::GetDataType<T>(),
                   output,
                   outputNumElements,
                   betann::GetDataType<U>(),
                   device_.CreateBufferFromVector(input),
                   input.size(),
                   shape,
                   strides,
                   axes);
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

TEST_F(ReduceTests, ReduceLast) {
  for (bool disableSubgroups : GetParameters()) {
    const uint32_t shapes[][2] = {
      {1, 1},
      {5, 5},
      {31, 15},
      {32, 16},
      {33, 17},
      {33, 600},
      {33, 800},
      {33, 1100},
    };
    for (auto [M, K] : shapes) {
      auto ints = RandomNumbers<int32_t>(M * K, 10);
      SCOPED_TRACE(fmt::format("Subgroups: {}, Shape: {}x{}",
                               !disableSubgroups, M, K));
      EXPECT_EQ(RunReduceLast<int32_t>(betann::ReduceType::Sum,
                                       ints, {M, K}, {1},
                                       disableSubgroups),
                Sum(ints, {M, K}, {K, 1}, {1}));
    }
  }
}

TEST_F(ReduceTests, ReduceRow) {
  for (bool disableSubgroups : GetParameters()) {
    const std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> shapes[] = {
      {{31, 65}, {1}},
      {{31, 65}, {0, 1}},
      {{31, 127}, {1}},
      {{31, 127}, {0, 1}},
      {{32, 128}, {1}},
      {{32, 128}, {0, 1}},
      {{33, 129}, {1}},
      {{33, 129}, {0, 1}},
      {{33, 520}, {1}},
      {{33, 1100}, {1}},
      {{31, 127, 127}, {2}},
      {{31, 127, 127}, {1, 2}},
      {{33, 33, 33, 4}, {3}},
      {{33, 33, 33, 4}, {2, 3}},
      {{33, 33, 33, 4}, {1, 2, 3}},
      {{7, 7, 7, 7, 7, 7}, {1, 2, 3, 4, 5}},
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
}

TEST_F(ReduceTests, ReduceNone) {
  const uint32_t sizes[] = {1, 31, 32, 33, 127, 128, 129};
  for (uint32_t size : sizes) {
    EXPECT_EQ(RunReduceNone<int32_t>(betann::ReduceType::Sum, size),
              (std::vector<int32_t>(size, 0)));
    EXPECT_EQ(RunReduceNone<uint32_t>(betann::ReduceType::Prod, size),
              (std::vector<uint32_t>(size, 1)));
    EXPECT_EQ(RunReduceNone<uint32_t>(betann::ReduceType::And, size),
              (std::vector<uint32_t>(size, true)));
    if (device_.SupportsF16()) {
      EXPECT_EQ(RunReduceNone<uint16_t>(betann::ReduceType::Prod, size),
                (std::vector<uint16_t>(size, betann::Float32ToFloat16(1))));
    }
  }
}

TEST_F(ReduceTests, Reduce) {
  EXPECT_EQ(RunReduce<float>(betann::ReductionPlanType::ReduceAll,
                             betann::ReduceType::Prod,
                             std::vector<int32_t>(), {}, {}, {}),
            (std::vector<float>{1}));
  auto a = RandomNumbers<int32_t>(10);
  EXPECT_EQ(RunReduce<float>(betann::ReductionPlanType::ReduceAll,
                             betann::ReduceType::Sum,
                             a, {}, {}, {}),
            (std::vector<float>{std::accumulate(a.begin(), a.end(), 0.f)}));
  EXPECT_EQ(RunReduce<int32_t>(betann::ReductionPlanType::ReduceRow,
                               betann::ReduceType::Sum,
                               a, {10}, {1}, {0}),
            Sum(a, {10}, {1}, {0}));
  EXPECT_EQ(RunReduce<int32_t>(betann::ReductionPlanType::ReduceRow,
                               betann::ReduceType::Sum,
                               a, {2, 5}, {5, 1}, {1}),
            Sum(a, {2, 5}, {5, 1}, {1}));
}
