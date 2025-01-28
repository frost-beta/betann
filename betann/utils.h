#ifndef BETANN_UTILS_H_
#define BETANN_UTILS_H_

#include <algorithm>
#include <numeric>

namespace betann {

template<typename T>
inline T NumElements(const std::vector<T>& shape) {
  return std::accumulate(shape.begin(), shape.end(),
                         1,
                         std::multiplies<T>());
}

}  // namespace betann

#endif  // BETANN_UTILS_H_
