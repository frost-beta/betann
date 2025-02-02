// Copyright 2017 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "betann/math.h"

#include <cmath>

namespace betann {

uint16_t Float32ToFloat16(float fp32) {
  uint32_t fp32i = BitCast<uint32_t>(fp32);
  uint32_t sign16 = (fp32i & 0x80000000) >> 16;
  uint32_t mantissaAndExponent = fp32i & 0x7FFFFFFF;
  if (mantissaAndExponent > 0x7F800000) {  // NaN
    return 0x7FFF;
  } else if (mantissaAndExponent > 0x47FFEFFF) {  // Infinity
    return static_cast<uint16_t>(sign16 | 0x7C00);
  } else if (mantissaAndExponent < 0x38800000) {  // Denormal
    uint32_t mantissa = (mantissaAndExponent & 0x007FFFFF) | 0x00800000;
    int32_t exponent = 113 - (mantissaAndExponent >> 23);
    if (exponent < 24) {
      mantissaAndExponent = mantissa >> exponent;
    } else {
      mantissaAndExponent = 0;
    }
    return static_cast<uint16_t>(
        sign16 | (mantissaAndExponent + 0x00000FFF
                                      + ((mantissaAndExponent >> 13) & 1))
                     >> 13);
  } else {
    return static_cast<uint16_t>(
        sign16 | (mantissaAndExponent + 0xC8000000
                                      + 0x00000FFF
                                      + ((mantissaAndExponent >> 13) & 1))
                     >> 13);
  }
}

float Float16ToFloat32(uint16_t fp16) {
  uint32_t tmp = (fp16 & 0x7fff) << 13 | (fp16 & 0x8000) << 16;
  float tmp2 = *reinterpret_cast<float*>(&tmp);
  return pow(2, 127 - 15) * tmp2;
}

bool IsFloat16NaN(uint16_t fp16) {
  return (fp16 & 0x7FFF) > 0x7C00;
}

}  // namespace betann
