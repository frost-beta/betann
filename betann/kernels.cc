#include "betann/kernels.h"

#include <absl/strings/substitute.h>
#include <fmt/format.h>

#include "betann/utils.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

template<typename... Args>
inline std::string GetShaderSource(Args&&... args) {
  return absl::Substitute(std::forward<Args>(args)...);
}

template<typename... Args>
inline std::string Append(std::string prefix, Args&&... args) {
  ((prefix += args), ...);
  return prefix;
}

Dims3 GetWorkgroupsCountContiguous(uint32_t numElements,
                                   uint32_t threadsPerDim,
                                   uint32_t workgroupSize) {
  Dims3 workgroupsCount;
  if (numElements > threadsPerDim) {
    workgroupsCount.x = DivFloor(threadsPerDim, workgroupSize);
    workgroupsCount.y = DivCeil(numElements, workgroupsCount.x);
  } else {
    workgroupsCount.y = DivCeil(numElements, workgroupSize);
  }
  return workgroupsCount;
}

Dims3 GetWorkgroupsCountGeneral(const std::vector<uint32_t>& shape,
                                uint32_t workgroupSize,
                                uint32_t workPerThread) {
  size_t numElements = NumElements(shape);
  size_t ndim = shape.size();
  size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
  size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
  size_t rest = numElements / (dim0 * dim1);
  if (ndim > 3) {
    dim0 = (dim0 + workPerThread - 1) / workPerThread;
  }
  Dims3 workgroupsCount;
  workgroupsCount.x = DivCeil(dim0, workgroupSize);
  workgroupsCount.y = DivCeil(dim1, workgroupSize);
  workgroupsCount.z = DivCeil(rest, workgroupSize);
  return workgroupsCount;
}

template<typename F>
void RunKernel(Device& device,
               const std::string& kernelName,
               const std::string& shaderSuffix,
               F&& getSource,
               std::initializer_list<wgpu::Buffer> buffers,
               Dims3 workgroupsCount) {
  const wgpu::ShaderModule& shader = device.CreateShaderModule(
      fmt::format("{}_{}", kernelName, shaderSuffix).c_str(),
      std::forward<F>(getSource));
  const wgpu::ComputePipeline& kernel = device.CreateKernel(
      shader,
      kernelName.c_str());
  device.RunKernel(kernel,
                   device.CreateBindGroup(kernel, std::move(buffers)),
                   workgroupsCount);
}

}  // namespace

void BinaryOpContiguous(Device& device,
                        const char* name,
                        BinaryOpType type,
                        const char* outputDataType,
                        const wgpu::Buffer& output,
                        uint32_t outputNumElements,
                        const char* inputDataType,
                        const wgpu::Buffer& a,
                        const wgpu::Buffer& b) {
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension * workgroupSize;
  bool use2DGrid = outputNumElements > maxThreadsPerGridDim;
  const char* typeStr = nullptr;
  switch (type) {
    case BinaryOpType::ScalarScalar:
      typeStr = "ss";
      break;
    case BinaryOpType::ScalarVector:
      typeStr = use2DGrid ? "sv2" : "sv";
      break;
    case BinaryOpType::VectorScalar:
      typeStr = use2DGrid ? "vs2" : "vs";
      break;
    case BinaryOpType::VectorVector:
      typeStr = use2DGrid ? "vv2" : "vv";
      break;
  }
  RunKernel(device,
            fmt::format("binary_{}_{}", typeStr, name),
            fmt::format("{}_{}", outputDataType, inputDataType),
            [&]() {
              return Append(GetShaderSource(wgsl_source_binary_contiguous,
                                            outputDataType,
                                            inputDataType,
                                            name),
                            wgsl_source_binary_ops);
            },
            {output, a, b},
            GetWorkgroupsCountContiguous(outputNumElements,
                                         maxThreadsPerGridDim,
                                         workgroupSize));
}

void BinaryOpGeneral(Device& device,
                     const char* name,
                     const char* outputDataType,
                     const wgpu::Buffer& output,
                     const std::vector<uint32_t>& shape,
                     const char* inputDataType,
                     const wgpu::Buffer& a,
                     size_t aNumElements,
                     const std::vector<uint32_t>& aStrides,
                     const wgpu::Buffer& b,
                     size_t bNumElements,
                     const std::vector<uint32_t>& bStrides) {
  if (aNumElements > UINT32_MAX || bNumElements > UINT32_MAX) {
    throw std::runtime_error(
        fmt::format("Number of elements ({}, {}) exceeds maximum index.",
                    aNumElements, bNumElements));
  }
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            shape.size() > 3
                ? fmt::format("binary_g_n{}_{}", workPerThread, name)
                : fmt::format("binary_g{}_{}", shape.size(), name),
            fmt::format("{}_{}", outputDataType, inputDataType),
            [&]() {
              return Append(GetShaderSource(wgsl_source_binary_general,
                                            outputDataType,
                                            inputDataType,
                                            name),
                            wgsl_source_binary_ops);
            },
            {
              output,
              device.CreateBufferFromVector(shape),
              a,
              device.CreateBufferFromVector(aStrides),
              b,
              device.CreateBufferFromVector(bStrides),
            },
            GetWorkgroupsCountGeneral(shape, workgroupSize, workPerThread));
}

void CopyContiguous(Device& device,
                    CopyType type,
                    const char* dstDataType,
                    const wgpu::Buffer& dst,
                    uint32_t dstNumElements,
                    const char* srcDataType,
                    const wgpu::Buffer& src) {
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension * workgroupSize;
  bool use2DGrid = dstNumElements > maxThreadsPerGridDim;
  const char* typeStr = nullptr;
  switch (type) {
    case CopyType::Scalar:
      typeStr = use2DGrid ? "s2" : "s";
      break;
    case CopyType::Vector:
      typeStr = use2DGrid ? "v2" : "v";
      break;
  }
  RunKernel(device,
            fmt::format("copy_{}", typeStr),
            fmt::format("{}_{}", dstDataType, srcDataType),
            [&]() {
              return GetShaderSource(wgsl_source_copy_contiguous,
                                     dstDataType,
                                     srcDataType);
            },
            {dst, src},
            GetWorkgroupsCountContiguous(dstNumElements,
                                         maxThreadsPerGridDim,
                                         workgroupSize));
}

void CopyGeneral(Device& device,
                 const char* dstDataType,
                 const wgpu::Buffer& dst,
                 const char* srcDataType,
                 const wgpu::Buffer& src,
                 const std::vector<uint32_t>& srcShape,
                 const std::vector<uint32_t>& srcStrides) {
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            srcShape.size() > 3
                ? fmt::format("copy_g_n{}", workPerThread)
                : fmt::format("copy_g{}", srcShape.size()),
            fmt::format("{}_{}", dstDataType, srcDataType),
            [&]() {
              return GetShaderSource(wgsl_source_copy_general,
                                     dstDataType,
                                     srcDataType);
            },
            {
              dst,
              src,
              device.CreateBufferFromVector(srcShape),
              device.CreateBufferFromVector(srcStrides),
            },
            GetWorkgroupsCountGeneral(srcShape, workgroupSize, workPerThread));
}

void CopyGeneralBoth(Device& device,
                     const char* dstDataType,
                     const wgpu::Buffer& dst,
                     const std::vector<uint32_t>& dstStrides,
                     const char* srcDataType,
                     const wgpu::Buffer& src,
                     const std::vector<uint32_t>& srcShape,
                     const std::vector<uint32_t>& srcStrides) {
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            srcShape.size() > 3
                ? fmt::format("copy_gg_n{}", workPerThread)
                : fmt::format("copy_gg{}", srcShape.size()),
            fmt::format("{}_{}", dstDataType, srcDataType),
            [&]() {
              return GetShaderSource(wgsl_source_copy_general_both,
                                     dstDataType,
                                     srcDataType);
            },
            {
              dst,
              device.CreateBufferFromVector(dstStrides),
              src,
              device.CreateBufferFromVector(srcShape),
              device.CreateBufferFromVector(srcStrides),
            },
            GetWorkgroupsCountGeneral(srcShape, workgroupSize, workPerThread));
}

void RandomBitsContiguous(Device& device,
                          DataType outDataType,
                          const wgpu::Buffer& out,
                          uint32_t outNumElements,
                          const wgpu::Buffer& keys,
                          uint32_t keysNumElements) {
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  uint32_t numKeys = keysNumElements / 2;  // each key consists of 2 items
  uint32_t bytesPerkey = outNumElements * SizeOf(outDataType) / numKeys;
  uint32_t outPerKey = DivCeil(bytesPerkey, 4u);
  Dims3 workgroupsCount;
  workgroupsCount.x = DivCeil(numKeys, workgroupSize);
  workgroupsCount.y = DivCeil(outPerKey / 2 + (outPerKey % 2), workgroupSize);
  RunKernel(device,
            "rbits_c",
            "",
            [&]() {
              return Append(GetShaderSource(wgsl_source_random_contiguous),
                            wgsl_source_random_utils);
            },
            {
              out,
              keys,
              device.CreateBufferFromData(&bytesPerkey, sizeof(uint32_t)),
            },
            workgroupsCount);
}

}  // namespace betann
