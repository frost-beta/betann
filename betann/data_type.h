#ifndef BETANN_DATA_TYPE_H_
#define BETANN_DATA_TYPE_H_

#include <stdexcept>

namespace betann {

enum class DataType {
  Bool,
  I32,
  U32,
  F32,
  F16,
};

constexpr size_t SizeOf(DataType dataType) {
  if (dataType == DataType::F16)
    return 2;
  return 4;
}

constexpr bool IsFloating(DataType dataType) {
  return dataType == DataType::F32 || dataType == DataType::F16;
}

constexpr bool IsUnsigned(DataType dataType) {
  return dataType == DataType::U32;
}

constexpr const char* WgslType(DataType dataType) {
  switch (dataType) {
    case DataType::Bool:
      return "bool";
    case DataType::I32:
      return "i32";
    case DataType::U32:
      return "u32";
    case DataType::F32:
      return "f32";
    case DataType::F16:
      return "f16";
  }
}

template<typename T>
inline DataType GetDataType() {
  throw std::runtime_error("Unsupported C++ data type in WebGPU.");
}
template<> inline DataType GetDataType<bool>() { return DataType::Bool; }
template<> inline DataType GetDataType<int32_t>() { return DataType::I32; }
template<> inline DataType GetDataType<uint32_t>() { return DataType::U32; }
template<> inline DataType GetDataType<float>() { return DataType::F32; }
template<> inline DataType GetDataType<uint16_t>() { return DataType::F16; }

}  // namespace betann

#endif  // BETANN_DATA_TYPE_H_
