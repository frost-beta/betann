#ifndef BETANN_UTILS_H_
#define BETANN_UTILS_H_

#include "betann/math.h"

#include <algorithm>
#include <numeric>
#include <vector>

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

}  // namespace betann

#endif  // BETANN_UTILS_H_
