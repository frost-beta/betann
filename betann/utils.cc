// Copyright © 2023-2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "betann/utils.h"

namespace betann {

std::tuple<std::vector<uint32_t>,
           std::vector<std::vector<uint32_t>>>
CollapseContiguousDimsImpl(const std::vector<uint32_t>& shape,
                           const std::vector<std::vector<uint32_t>>& strides,
                           int64_t sizeCap) {
  // Make a vector that has axes separated with -1. Collapse all axes between
  // -1.
  std::vector<int64_t> toCollapse;
  if (shape.size() > 0) {
    if (shape[0] != 1) {
      toCollapse.push_back(0);
    }
    size_t size = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
      bool contiguous = true;
      size *= shape[i];
      for (const auto& st : strides) {
        if (st[i] * shape[i] != st[i - 1] || size > sizeCap) {
          contiguous = false;
          size = shape[i];
          break;
        }
      }
      if (!contiguous) {
        toCollapse.push_back(-1);
      }
      if (shape[i] != 1) {
        toCollapse.push_back(i);
      }
    }
    toCollapse.push_back(-1);
  }

  std::vector<uint32_t> outShape;
  std::vector<std::vector<uint32_t>> outStrides(strides.size());
  for (size_t i = 0;;) {
    while (i < toCollapse.size() && toCollapse[i] == -1) {
      ++i;
    };
    if (i == toCollapse.size()) {
      break;
    }
    uint32_t currentShape = shape[toCollapse[i]];
    size_t k = i;
    while (toCollapse[++k] != -1) {
      currentShape *= shape[toCollapse[k]];
    }
    outShape.push_back(currentShape);
    for (size_t j = 0; j < strides.size(); j++) {
      const auto& st = strides[j];
      outStrides[j].push_back(st[toCollapse[k - 1]]);
    }
    i = k + 1;
  }

  if (!shape.empty() && outShape.empty()) {
    outShape.push_back(1);
    for (auto& outStride : outStrides) {
      outStride.push_back(0);
    }
  }
  return {std::move(outShape), std::move(outStrides)};
}

std::tuple<std::vector<uint32_t>,
           std::vector<uint32_t>>
CollapseContiguousDimsImpl(const std::vector<uint32_t>& shape,
                           const std::vector<uint32_t>& strides,
                           int64_t sizeCap) {
  std::vector<uint32_t> collapsedShape;
  std::vector<uint32_t> collapsedStrides;

  if (shape.size() > 0) {
    collapsedShape.push_back(shape[0]);
    collapsedStrides.push_back(strides[0]);
    for (size_t i = 1; i < shape.size(); i++) {
      if (shape[i] == 1) {
        continue;
      } else if (strides[i] * shape[i] != collapsedStrides.back() ||
                 collapsedShape.back() * shape[i] > sizeCap) {
        collapsedShape.push_back(shape[i]);
        collapsedStrides.push_back(strides[i]);
      } else {
        collapsedShape.back() *= shape[i];
        collapsedStrides.back() = strides[i];
      }
    }
  }

  return {std::move(collapsedShape), std::move(collapsedStrides)};
}

}  // namespace betann
