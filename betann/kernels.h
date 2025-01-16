#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <string>

#include "betann/device.h"

namespace betann {

void RunBinaryOp(Device& device,
                 const char* type,
                 const char* name,
                 size_t numElements,
                 const char* inputDType,
                 const wgpu::Buffer& a,
                 const wgpu::Buffer& b,
                 const char* outputDType,
                 const wgpu::Buffer& output);

std::string GetBinaryShaderSource(const char* op,
                                  const char* inputDType,
                                  const char* outputDType);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
