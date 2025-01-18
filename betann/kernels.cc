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

std::string GetShaderSourceBinary(const char* op,
                                  const char* inputDataType,
                                  const char* outputDataType) {
  std::string preprocessed = absl::Substitute(wgsl_source_binary,
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

void BinaryOp(Device& device,
              BinaryOpType type,
              const char* name,
              size_t outputSize,
              const char* outputDataType,
              const wgpu::Buffer& output,
              const char* inputDataType,
              const wgpu::Buffer& a,
              const wgpu::Buffer& b) {
  if (outputSize > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}) exceeds maximum value.",
                    outputSize));
  }
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension;
  bool use2DGrid = outputSize > maxThreadsPerGridDim;
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
        return GetShaderSourceBinary(name, inputDataType, outputDataType);
      });
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  if (use2DGrid) {
    gridDims.x =
        std::floor(maxThreadsPerGridDim / static_cast<float>(workgroupSize));
    gridDims.y = std::ceil(outputSize / static_cast<float>(gridDims.x));
  } else {
    gridDims.x = std::ceil(outputSize / static_cast<float>(workgroupSize));
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
  size_t outputSize = std::accumulate(shape.begin(), shape.end(),
                                      1,
                                      std::multiplies<uint32_t>());
  if (aNumElements > INT32_MAX ||
      bNumElements > INT32_MAX ||
      outputSize > INT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}, {}, {}) exceeds maximum index.",
                    outputSize, aNumElements, bNumElements));
  }
  size_t ndim = shape.size();
  if (ndim > 3) {
    throw std::runtime_error("BinaryOpGeneral supports atmost 3 dimensions.");
  }
  std::string kernelName = fmt::format("binary_g{}_{}", ndim, name);
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
         device.CreateBufferFromData(wgpu::BufferUsage::Storage,
                                     aStrides.size() * sizeof(uint32_t),
                                     aStrides.data()),
         b,
         device.CreateBufferFromData(wgpu::BufferUsage::Storage,
                                     bStrides.size() * sizeof(uint32_t),
                                     bStrides.data()),
         device.CreateBufferFromData(wgpu::BufferUsage::Storage,
                                     shape.size() * sizeof(uint32_t),
                                     shape.data()),
         output,
      });
  bindGroup.SetLabel(kernelName.c_str());
  size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
  size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
  size_t rest = outputSize / (dim0 * dim1);
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  GridDims gridDims;
  gridDims.x = std::ceil(dim0 / static_cast<float>(workgroupSize));
  gridDims.y = std::ceil(dim1 / static_cast<float>(workgroupSize));
  gridDims.z = std::ceil(rest / static_cast<float>(workgroupSize));
  device.RunKernel(kernel, bindGroup, gridDims);
}

}  // namespace betann
