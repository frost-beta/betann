#ifndef BETANN_BUFFER_H_
#define BETANN_BUFFER_H_

#include <webgpu/webgpu_cpp.h>

namespace betann {

using BufferUsage = wgpu::BufferUsage;

struct Buffer {
  wgpu::Buffer data;
  uint64_t size = WGPU_WHOLE_SIZE;
  uint64_t offset = 0;

  Buffer(wgpu::Buffer data) : data(std::move(data)) {}
  Buffer(std::nullptr_t) : data(nullptr) {}

  uint64_t GetSize() const {
    return size == WGPU_WHOLE_SIZE ? data.GetSize() : size;
  }
};

}  // namespace betann

#endif  // BETANN_BUFFER_H_
