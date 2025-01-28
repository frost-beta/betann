#include "betann/kernels.h"

#include <absl/strings/substitute.h>
#include <fmt/format.h>

#include "betann/utils.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

inline const char* GetBinaryOpTypeStr(BinaryOpType type, bool largeArray) {
  switch (type) {
    case BinaryOpType::ScalarScalar:
      return "ss";
    case BinaryOpType::ScalarVector:
      return largeArray ? "sv2" : "sv";
    case BinaryOpType::VectorScalar:
      return largeArray ? "vs2" : "vs";
    case BinaryOpType::VectorVector:
      return largeArray ? "vv2" : "vv";
  }
}

inline const char* GetCopyTypeStr(CopyType type, bool largeArray) {
  switch (type) {
    case CopyType::Scalar:
      return largeArray ? "s2" : "s";
    case CopyType::Vector:
      return largeArray ? "v2" : "v";
  }
}

template<typename... Args>
inline std::string GetShaderSource(Args&&... args) {
  return absl::Substitute(std::forward<Args>(args)...);
}

template<typename... Args>
inline std::string Append(std::string prefix, Args&&... args) {
  ((prefix += args), ...);
  return prefix;
}

template<typename F>
void RunKernel(Device& device,
               const std::string& kernelName,
               const std::string& shaderSuffix,
               F&& getSource,
               std::initializer_list<wgpu::Buffer> buffers,
               GridDims gridDims) {
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      fmt::format("{}_{}", kernelName, shaderSuffix).c_str(),
      std::forward<F>(getSource));
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, std::move(buffers)),
                   gridDims);
}

}  // namespace

void BinaryOpContiguous(Device& device,
                        const char* name,
                        BinaryOpType type,
                        const char* outputDataType,
                        const wgpu::Buffer& output,
                        size_t outputNumElements,
                        const char* inputDataType,
                        const wgpu::Buffer& a,
                        const wgpu::Buffer& b) {
  if (outputNumElements > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}) exceeds maximum index.",
                    outputNumElements));
  }
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension;
  bool use2DGrid = outputNumElements > maxThreadsPerGridDim;
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  if (use2DGrid) {
    gridDims.x =
        std::floor(maxThreadsPerGridDim / static_cast<float>(workgroupSize));
    gridDims.y = std::ceil(outputNumElements / static_cast<float>(gridDims.x));
  } else {
    gridDims.x = std::ceil(outputNumElements / static_cast<float>(workgroupSize));
  }
  RunKernel(device,
            fmt::format("binary_{}_{}",
                        GetBinaryOpTypeStr(type, use2DGrid),
                        name),
            fmt::format("{}_{}", outputDataType, inputDataType),
            [&]() {
              return Append(GetShaderSource(wgsl_source_binary_contiguous,
                                            outputDataType,
                                            inputDataType,
                                            name),
                            wgsl_source_binary_ops);
            },
            {output, a, b},
            gridDims);
}

void BinaryOpGeneral(Device& device,
                     const char* name,
                     const std::vector<uint32_t>& shape,
                     const char* outputDataType,
                     const wgpu::Buffer& output,
                     const char* inputDataType,
                     const wgpu::Buffer& a,
                     size_t aNumElements,
                     const std::vector<uint32_t>& aStrides,
                     const wgpu::Buffer& b,
                     size_t bNumElements,
                     const std::vector<uint32_t>& bStrides) {
  size_t outputNumElements = NumElements(shape);
  if (aNumElements > UINT32_MAX ||
      bNumElements > UINT32_MAX ||
      outputNumElements > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}, {}, {}) exceeds maximum index.",
                    outputNumElements, aNumElements, bNumElements));
  }
  size_t ndim = shape.size();
  size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
  size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
  size_t rest = outputNumElements / (dim0 * dim1);
  if (ndim > 3) {
    const uint32_t workPerThread = 2;
    dim0 = (dim0 + workPerThread - 1) / workPerThread;
  }
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  gridDims.x = std::ceil(dim0 / static_cast<float>(workgroupSize));
  gridDims.y = std::ceil(dim1 / static_cast<float>(workgroupSize));
  gridDims.z = std::ceil(rest / static_cast<float>(workgroupSize));
  RunKernel(device,
            ndim > 3 ? fmt::format("binary_gn2_{}", name)
                     : fmt::format("binary_g{}_{}", ndim, name),
            fmt::format("{}_{}", outputDataType, inputDataType),
            [&]() {
              return Append(GetShaderSource(wgsl_source_binary_general,
                                            outputDataType,
                                            inputDataType,
                                            name),
                            wgsl_source_binary_ops);
            },
            {
              device.CreateBufferFromVector(shape),
              output,
              a,
              device.CreateBufferFromVector(aStrides),
              b,
              device.CreateBufferFromVector(bStrides),
            },
            gridDims);
}

void CopyContiguous(Device& device,
                    CopyType type,
                    const char* dstDataType,
                    const wgpu::Buffer& dst,
                    size_t dstNumElements,
                    const char* srcDataType,
                    const wgpu::Buffer& src) {
  if (dstNumElements > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}) exceeds maximum index.",
                    dstNumElements));
  }
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension;
  bool use2DGrid = dstNumElements > maxThreadsPerGridDim;
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  if (use2DGrid) {
    gridDims.x =
        std::floor(maxThreadsPerGridDim / static_cast<float>(workgroupSize));
    gridDims.y = std::ceil(dstNumElements / static_cast<float>(gridDims.x));
  } else {
    gridDims.x = std::ceil(dstNumElements / static_cast<float>(workgroupSize));
  }
  RunKernel(device,
            fmt::format("copy_{}", GetCopyTypeStr(type, use2DGrid)),
            fmt::format("{}_{}", dstDataType, srcDataType),
            [&]() {
              return GetShaderSource(wgsl_source_copy_contiguous,
                                     dstDataType,
                                     srcDataType);
            },
            {dst, src},
            gridDims);
}

}  // namespace betann
