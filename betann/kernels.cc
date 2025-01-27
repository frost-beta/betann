#include "betann/kernels.h"

#include <iostream>
#include <functional>
#include <numeric>

#include <absl/strings/substitute.h>
#include <fmt/format.h>

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

std::string GetShaderSourceBinaryContiguous(const char* op,
                                            const char* inputDataType,
                                            const char* outputDataType) {
  std::string preprocessed = absl::Substitute(wgsl_source_binary_contiguous,
                                              op,
                                              inputDataType,
                                              outputDataType);
  preprocessed += wgsl_source_binary_ops;
  return preprocessed;
}

std::string GetShaderSourceBinaryGeneral(const char* op,
                                         const char* inputDataType,
                                         const char* outputDataType) {
  std::string preprocessed = absl::Substitute(wgsl_source_binary_general,
                                              op,
                                              inputDataType,
                                              outputDataType);
  preprocessed += wgsl_source_binary_ops;
  return preprocessed;
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
  std::string kernelName = fmt::format("binary_{}_{}",
                                       GetBinaryOpTypeStr(type, use2DGrid),
                                       name);
  std::string shaderName = fmt::format("{}_{}_{}",
                                       kernelName,
                                       inputDataType,
                                       outputDataType);
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      shaderName.c_str(),
      [&]() {
        return GetShaderSourceBinaryContiguous(name,
                                               inputDataType,
                                               outputDataType);
      });
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  if (use2DGrid) {
    gridDims.x =
        std::floor(maxThreadsPerGridDim / static_cast<float>(workgroupSize));
    gridDims.y = std::ceil(outputNumElements / static_cast<float>(gridDims.x));
  } else {
    gridDims.x = std::ceil(outputNumElements / static_cast<float>(workgroupSize));
  }
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, {a, b, output}),
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
  size_t outputNumElements = std::accumulate(shape.begin(), shape.end(),
                                             1,
                                             std::multiplies<uint32_t>());
  if (aNumElements > UINT32_MAX ||
      bNumElements > UINT32_MAX ||
      outputNumElements > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}, {}, {}) exceeds maximum index.",
                    outputNumElements, aNumElements, bNumElements));
  }
  size_t ndim = shape.size();
  std::string kernelName;
  if (ndim > 3)
    kernelName = fmt::format("binary_gn2_{}", name);
  else
    kernelName = fmt::format("binary_g{}_{}", ndim, name);
  std::string shaderName = fmt::format("{}_{}_{}",
                                       kernelName,
                                       inputDataType,
                                       outputDataType);
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      shaderName.c_str(),
      [&]() {
        return GetShaderSourceBinaryGeneral(name,
                                            inputDataType,
                                            outputDataType);
      });
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  wgpu::BindGroup bindGroup = device.CreateBindGroup(
      kernel,
      {
         a,
         device.CreateBufferFromVector(aStrides),
         b,
         device.CreateBufferFromVector(bStrides),
         device.CreateBufferFromVector(shape),
         output,
      });
  bindGroup.SetLabel(kernelName.c_str());
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
  device.RunKernel(kernel, bindGroup, gridDims);
}

}  // namespace betann
