#ifndef BETANN_MATH_H_
#define BETANN_MATH_H_

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace betann {

template<typename T, typename U>
inline std::enable_if_t<std::is_unsigned_v<T> && std::is_unsigned_v<U>, T>
DivCeil(T a, U b) {
  return (a + (b - 1)) / b;
}

template<typename T, typename U>
inline std::enable_if_t<std::is_unsigned_v<T> && std::is_unsigned_v<U>, T>
DivFloor(T a, U b) {
  return a / b;
}

template <typename U, typename T>
inline U BitCast(const T& source) {
  static_assert(sizeof(U) == sizeof(T), "BitCast: cannot lose precision.");
  U output;
  std::memcpy(&output, &source, sizeof(U));
  return output;
}

uint16_t Float32ToFloat16(float fp32);
float Float16ToFloat32(uint16_t fp16);
bool IsFloat16NaN(uint16_t fp16);

}  // namespace betann

#endif  // BETANN_MATH_H_
