#include "betann/kernels.h"

#include "absl/strings/substitute.h"
#include "wgsl_sources.h"

namespace betann {

std::string GetBinaryShaderSource(const char* op,
                                  const char* inputDType,
                                  const char* outputDType) {
  constexpr std::string_view source(&wgsl_source_binary.front(),
                                    wgsl_source_binary.size());
  return absl::Substitute(source, op, inputDType, outputDType);
}

}  // namespace betann
