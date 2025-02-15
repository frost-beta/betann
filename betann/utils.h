#ifndef BETANN_UTILS_H_
#define BETANN_UTILS_H_

#include <algorithm>
#include <climits>
#include <numeric>
#include <vector>

#include "betann/math.h"

namespace betann {

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

// Slice a vector.
template<typename T>
inline std::vector<T> Slice(const std::vector<T>& vec,
                            int start = 0,
                            int end = -1) {
  if (start < 0)
    start += static_cast<int>(vec.size());
  if (end < 0)
    end += static_cast<int>(vec.size());
  if (start >= vec.size() || end > vec.size())
    return {};
  return std::vector<T>(vec.begin() + start, vec.begin() + end);
}

template<typename T, typename V, size_t... Is>
auto VectorToTuple(T first, V vec, std::index_sequence<Is...>) {
    return std::make_tuple(std::move(first), std::move(vec[Is])...);
}

std::tuple<std::vector<uint32_t>,
           std::vector<std::vector<uint32_t>>>
CollapseContiguousDims(const std::vector<uint32_t>& shape,
                       const std::vector<std::vector<uint32_t>>& strides,
                       int64_t sizeCap = INT_MAX);

struct DisableCollapseDims {
  DisableCollapseDims() { isDisabled = true; }
  ~DisableCollapseDims() { isDisabled = false; }
  static bool isDisabled;
};

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
  if (DisableCollapseDims::isDisabled)
    return {shape, args...};
  auto [outShape, outStrides] = CollapseContiguousDims(shape, {args...});
  return VectorToTuple(std::move(outShape),
                       std::move(outStrides),
                       std::make_index_sequence<sizeof...(Args)>());
}

}  // namespace betann

#endif  // BETANN_UTILS_H_
