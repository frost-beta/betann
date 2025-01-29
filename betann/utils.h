#ifndef BETANN_UTILS_H_
#define BETANN_UTILS_H_

#include <algorithm>
#include <numeric>
#include <vector>

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
