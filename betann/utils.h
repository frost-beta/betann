#ifndef BETANN_UTILS_H_
#define BETANN_UTILS_H_

#include <algorithm>
#include <climits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "betann/math.h"

namespace betann {

enum class DataType {
  bool_,
  i32,
  u32,
  f32,
  f16,
};

constexpr size_t SizeOf(DataType dataType) {
  switch (dataType) {
    case DataType::bool_:
    case DataType::i32:
    case DataType::u32:
    case DataType::f32:
      return 4;
    case DataType::f16:
      return 2;
  }
}

constexpr bool IsFloating(DataType dataType) {
  return dataType == DataType::f32 || dataType == DataType::f16;
}

constexpr bool IsUnsigned(DataType dataType) {
  return dataType == DataType::u32;
}

constexpr const char* WgslType(DataType dataType) {
  switch (dataType) {
    case DataType::bool_:
      return "bool";
    case DataType::i32:
      return "i32";
    case DataType::u32:
      return "u32";
    case DataType::f32:
      return "f32";
    case DataType::f16:
      return "f16";
  }
}

template<typename T>
inline DataType GetDataType() {
  throw std::runtime_error("Unsupported C++ data type in WebGPU.");
}
template<> inline DataType GetDataType<bool>() { return DataType::bool_; }
template<> inline DataType GetDataType<int32_t>() { return DataType::i32; }
template<> inline DataType GetDataType<uint32_t>() { return DataType::u32; }
template<> inline DataType GetDataType<float>() { return DataType::f32; }
template<> inline DataType GetDataType<uint16_t>() { return DataType::f16; }

inline uint32_t NumElements(const std::vector<uint32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(),
                         1,
                         std::multiplies<uint32_t>());
}

inline uint32_t NumElements(const std::vector<uint32_t>& shape,
                            const std::vector<uint32_t>& strides) {
  uint32_t offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    offset += (shape[i] - 1) * strides[i];
  }
  return offset + 1;
}

std::tuple<std::vector<uint32_t>,
           std::vector<std::vector<uint32_t>>>
CollapseContiguousDimsImpl(const std::vector<uint32_t>& shape,
                           const std::vector<std::vector<uint32_t>>& strides,
                           int64_t sizeCap = INT_MAX);
std::tuple<std::vector<uint32_t>,
           std::vector<uint32_t>>
CollapseContiguousDimsImpl(const std::vector<uint32_t>& shape,
                           const std::vector<uint32_t>& strides,
                           int64_t sizeCap = INT_MAX);

template<typename T, typename V, size_t... Is>
auto VectorToTuple(T first, V vec, std::index_sequence<Is...>) {
    return std::make_tuple(std::move(first), std::move(vec[Is])...);
}

// Collapse dims that are contiguous to possibly route to a better kernel.
// e.g. for x = transpose(array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2}), {2, 0, 1})
// should return {{2, 4}, {{1, 2}}}.
//
// When multiple arrays are passed they should all have the same shape. The
// collapsed axes are also the same so one shape is returned.
template<typename... Args>
inline std::tuple<std::vector<uint32_t>, Args...> CollapseContiguousDims(
    const std::vector<uint32_t>& shape,
    const Args&... args) {
  if constexpr (sizeof...(Args) == 1) {
    return CollapseContiguousDimsImpl(shape, args...);
  } else {
    auto [outShape, outStrides] = CollapseContiguousDimsImpl(shape, {args...});
    return VectorToTuple(std::move(outShape),
                         std::move(outStrides),
                         std::make_index_sequence<sizeof...(Args)>());
  }
}

}  // namespace betann

#endif  // BETANN_UTILS_H_
