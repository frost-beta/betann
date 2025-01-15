#ifndef BETANN_BINARY_H_
#define BETANN_BINARY_H_

#include "betann/device.h"

namespace betann {

template<typename T>
void BinaryOp(Device& device,
              const char* name,
              const wgpu::Buffer& a,
              const wgpu::Buffer& b,
              const wgpu::Buffer& c) {
}

}  // namespace betann

#endif  // BETANN_BINARY_H_
