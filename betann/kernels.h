#ifndef BETANN_KERNELS_H_
#define BETANN_KERNELS_H_

#include <string>

namespace betann {

std::string GetBinaryShaderSource(const char* op,
                                  const char* inputDType,
                                  const char* outputDType);

}  // namespace betann

#endif  // BETANN_KERNELS_H_
