#include "betann/kernels.h"

#include <fmt/format.h>

#include "betann/kernels_helper.h"
#include "wgsl_sources.h"

namespace betann {

namespace {

struct Dims {
  uint32_t ndim;
  uint32_t dim0 = 1;
  uint32_t dim1 = 1;
  uint32_t rest = 1;
};

Dims GetDims(const std::vector<uint32_t>& shape) {
  Dims dims;
  dims.ndim = shape.size();
  dims.dim0 = dims.ndim > 0 ? shape[dims.ndim - 1] : 1;
  dims.dim1 = dims.ndim > 1 ? shape[dims.ndim - 2] : 1;
  if (dims.ndim > 2) {
    for (uint32_t d = 0; d < dims.ndim - 2; d++)
      dims.rest *= shape[d];
  }
  return dims;
}

Dims3 GetWorkgroupsCountContiguous(uint32_t numElements,
                                   uint32_t threadsPerDim,
                                   uint32_t workgroupSize) {
  Dims3 workgroupsCount;
  if (numElements > threadsPerDim) {
    workgroupsCount.x = DivFloor(threadsPerDim, workgroupSize);
    workgroupsCount.y = DivCeil(numElements, workgroupsCount.x);
  } else {
    workgroupsCount.x = DivCeil(numElements, workgroupSize);
  }
  return workgroupsCount;
}

Dims3 GetWorkgroupsCountGeneral(const std::vector<uint32_t>& shape,
                                uint32_t workgroupSize,
                                uint32_t workPerThread) {
  Dims dims = GetDims(shape);
  if (dims.ndim > 3)
    dims.dim0 = (dims.dim0 + workPerThread - 1) / workPerThread;
  Dims3 workgroupsCount;
  workgroupsCount.x = DivCeil(dims.dim0, workgroupSize);
  workgroupsCount.y = DivCeil(dims.dim1, workgroupSize);
  workgroupsCount.z = DivCeil(dims.rest, workgroupSize);
  return workgroupsCount;
}

}  // namespace

void ArrayRange(Device& device,
                double start,
                double step,
                DataType dataType,
                const Buffer& out) {
  const uint32_t workgroupSize = 64;
  uint32_t outNumElements = out.GetSize() / SizeOf(dataType);
  RunKernel(device,
            "arange",
            fmt::format("arange_{}", WgslType(dataType)),
            [&]() {
              return ParseTemplate(
                  wgsl_source_arange,
                  {
                    {"enable_f16", EnableF16(device, dataType)},
                    {"dtype", WgslType(dataType)},
                  });
            },
            {
              device.CreateBufferFromScalar(start, dataType),
              device.CreateBufferFromScalar(step, dataType),
              out,
            },
            {DivCeil(outNumElements, workgroupSize)});
}

void BinaryOpContiguous(Device& device,
                        const char* name,
                        BinaryOpType type,
                        DataType outputDataType,
                        const Buffer& output,
                        uint32_t outputNumElements,
                        DataType inputDataType,
                        const Buffer& a,
                        const Buffer& b) {
  const uint32_t workgroupSize = 64;  // TODO(zcbenz): make it dynamic
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
            fmt::format("binary_{}_{}_{}",
                        name,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_binary_contiguous,
                      {
                        {"enable_f16", EnableF16(device,
                                                 outputDataType,
                                                 inputDataType)},
                        {"output_dtype", WgslType(outputDataType)},
                        {"input_dtype", WgslType(inputDataType)},
                        {"op", name},
                      }),
                  ParseTemplate(
                      wgsl_source_binary_ops,
                      {
                        {"input_is_floating", IsFloating(inputDataType)},
                        {"input_is_integer", IsInteger(inputDataType)},
                      }));
            },
            {output, a, b},
            GetWorkgroupsCountContiguous(outputNumElements,
                                         maxThreadsPerGridDim,
                                         workgroupSize));
}

void BinaryOpGeneral(Device& device,
                     const char* name,
                     DataType outputDataType,
                     const Buffer& output,
                     const std::vector<uint32_t>& shapePre,
                     DataType inputDataType,
                     const Buffer& a,
                     const std::vector<uint32_t>& aStridesPre,
                     const Buffer& b,
                     const std::vector<uint32_t>& bStridesPre) {
  auto [shape, aStrides, bStrides] =
      CollapseContiguousDims(shapePre, aStridesPre, bStridesPre);
  if (shape.size() < 2)
    throw std::runtime_error("BinaryOpGeneral do not take contiguous inputs.");
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            shape.size() > 3
                ? fmt::format("binary_g_n{}_{}", workPerThread, name)
                : fmt::format("binary_g{}_{}", shape.size(), name),
            fmt::format("binary_g_{}_{}_{}",
                        name,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_binary_general,
                      {
                        {"enable_f16", EnableF16(device,
                                                 outputDataType,
                                                 inputDataType)},
                        {"output_dtype", WgslType(outputDataType)},
                        {"input_dtype", WgslType(inputDataType)},
                        {"op", name},
                      }),
                  ParseTemplate(
                      wgsl_source_binary_ops,
                      {
                        {"input_is_floating", IsFloating(inputDataType)},
                        {"input_is_integer", IsInteger(inputDataType)},
                      }),
                  wgsl_source_utils);
            },
            {
              output,
              device.CreateBufferFromVector(shape),
              a,
              device.CreateBufferFromVector(aStrides),
              b,
              device.CreateBufferFromVector(bStrides),
              shape.size() > 3
                  ? device.CreateBufferFromStruct(GetDims(shape))
                  : nullptr,
            },
            GetWorkgroupsCountGeneral(shape, workgroupSize, workPerThread));
}

void CopyContiguous(Device& device,
                    CopyType type,
                    DataType dstDataType,
                    const Buffer& dst,
                    uint32_t dstNumElements,
                    DataType srcDataType,
                    const Buffer& src) {
  const uint32_t workgroupSize = 64;  // TODO(zcbenz): make it dynamic
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
            fmt::format("copy_{}_{}",
                        WgslType(dstDataType),
                        WgslType(srcDataType)),
            [&]() {
              return ParseTemplate(
                  wgsl_source_copy_contiguous,
                  {
                    {"enable_f16", EnableF16(device, dstDataType, srcDataType)},
                    {"dst_dtype", WgslType(dstDataType)},
                    {"src_dtype", WgslType(srcDataType)},
                  });
            },
            {dst, src},
            GetWorkgroupsCountContiguous(dstNumElements,
                                         maxThreadsPerGridDim,
                                         workgroupSize));
}

void CopyGeneral(Device& device,
                 DataType dstDataType,
                 const Buffer& dst,
                 DataType srcDataType,
                 const Buffer& src,
                 const std::vector<uint32_t>& srcShapePre,
                 const std::vector<uint32_t>& srcStridesPre) {
  auto [srcShape, srcStrides] =
      CollapseContiguousDims(srcShapePre, srcStridesPre);
  if (srcShape.size() < 2)
    throw std::runtime_error("CopyGeneral do not take contiguous inputs.");
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            srcShape.size() > 3
                ? fmt::format("copy_g_n{}", workPerThread)
                : fmt::format("copy_g{}", srcShape.size()),
            fmt::format("copy_g_{}_{}",
                        WgslType(dstDataType),
                        WgslType(srcDataType)),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_copy_general,
                      {
                        {"enable_f16", EnableF16(device,
                                                 dstDataType,
                                                 srcDataType)},
                        {"dst_dtype", WgslType(dstDataType)},
                        {"src_dtype", WgslType(srcDataType)},
                      }),
                  wgsl_source_utils);
            },
            {
              dst,
              src,
              device.CreateBufferFromVector(srcShape),
              device.CreateBufferFromVector(srcStrides),
              srcShape.size() > 3
                  ? device.CreateBufferFromStruct(GetDims(srcShape))
                  : nullptr,
            },
            GetWorkgroupsCountGeneral(srcShape, workgroupSize, workPerThread));
}

void CopyGeneralBoth(Device& device,
                     DataType dstDataType,
                     const Buffer& dst,
                     const std::vector<uint32_t>& dstStridesPre,
                     DataType srcDataType,
                     const Buffer& src,
                     const std::vector<uint32_t>& srcShapePre,
                     const std::vector<uint32_t>& srcStridesPre) {
  auto [srcShape, srcStrides, dstStrides] =
      CollapseContiguousDims(srcShapePre, srcStridesPre, dstStridesPre);
  if (srcShape.size() < 2)
    throw std::runtime_error("CopyGeneralBoth do not take contiguous inputs.");
  const uint32_t workPerThread = 2;
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            srcShape.size() > 3
                ? fmt::format("copy_gg_n{}", workPerThread)
                : fmt::format("copy_gg{}", srcShape.size()),
            fmt::format("copy_gg_{}_{}",
                        WgslType(dstDataType),
                        WgslType(srcDataType)),
            [&]() {
              return Append(
                 ParseTemplate(
                     wgsl_source_copy_general_both,
                     {
                       {"enable_f16", EnableF16(device,
                                                dstDataType,
                                                srcDataType)},
                       {"dst_dtype", WgslType(dstDataType)},
                       {"src_dtype", WgslType(srcDataType)},
                     }),
                 wgsl_source_utils);
            },
            {
              dst,
              device.CreateBufferFromVector(dstStrides),
              src,
              device.CreateBufferFromVector(srcShape),
              device.CreateBufferFromVector(srcStrides),
              srcShape.size() > 3
                  ? device.CreateBufferFromStruct(GetDims(srcShape))
                  : nullptr,
            },
            GetWorkgroupsCountGeneral(srcShape, workgroupSize, workPerThread));
}

void RandomBitsContiguous(Device& device,
                          DataType outDataType,
                          const Buffer& out,
                          uint32_t outNumElements,
                          const Buffer& keys,
                          uint32_t keysNumElements) {
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  uint32_t numKeys = keysNumElements / 2;  // each key consists of 2 items
  uint32_t bytesPerkey = outNumElements * SizeOf(outDataType) / numKeys;
  uint32_t outPerKey = DivCeil(bytesPerkey, 4u);
  Dims3 workgroupsCount;
  workgroupsCount.x = DivCeil(numKeys, workgroupSize);
  workgroupsCount.y = DivCeil(outPerKey / 2 + (outPerKey % 2), workgroupSize);
  RunKernel(device,
            "rbits",
            "rbits",
            [&]() {
              return Append(ParseTemplate(wgsl_source_random,
                                          {{"contiguous", true}}),
                            wgsl_source_utils);
            },
            {
              out,
              device.CreateBufferFromScalar(bytesPerkey),
              keys,
            },
            workgroupsCount);
}

void RandomBitsGeneral(Device& device,
                       DataType outDataType,
                       const Buffer& out,
                       uint32_t outNumElements,
                       const Buffer& keys,
                       const std::vector<uint32_t>& keysShape,
                       const std::vector<uint32_t>& keysStrides) {
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  uint32_t numKeys = NumElements(keysShape) / 2;
  uint32_t bytesPerkey = outNumElements * SizeOf(outDataType) / numKeys;
  uint32_t outPerKey = DivCeil(bytesPerkey, 4u);
  Dims3 workgroupsCount;
  workgroupsCount.x = DivCeil(numKeys, workgroupSize);
  workgroupsCount.y = DivCeil(outPerKey / 2 + (outPerKey % 2), workgroupSize);
  RunKernel(device,
            "rbits",
            "rbits_g",
            [&]() {
              return Append(ParseTemplate(wgsl_source_random,
                                          {{"contiguous", false}}),
                            wgsl_source_utils);
            },
            {
              out,
              device.CreateBufferFromScalar(bytesPerkey),
              keys,
              device.CreateBufferFromVector(keysShape),
              device.CreateBufferFromVector(keysStrides),
            },
            workgroupsCount);
}

uint32_t SortBlockSize() {
  const uint32_t workgroupSize = 256;  // TODO(zcbenz): make it dynamic
  const uint32_t workPerThread = 8;
  return  workgroupSize * workPerThread;
}

void SortBlock(Device& device,
               uint32_t axis,
               SortInputType inputType,
               SortResultType resultType,
               const Buffer& out,
               const std::vector<uint32_t>& outStrides,
               DataType inputDataType,
               const Buffer& input,
               const std::vector<uint32_t>& inputShape,
               const std::vector<uint32_t>& inputStrides) {
  uint32_t sizeSortedAxis = inputShape[axis];
  if (sizeSortedAxis > SortBlockSize()) {
    throw std::runtime_error(
        fmt::format("Elements number of sorted axis ({}) exceeds limit ({}).",
                    sizeSortedAxis, SortBlockSize()));
  }
  auto removeAxis = [](const std::vector<uint32_t>& input, uint32_t axis) {
    auto ret = input;
    ret.erase(ret.begin() + axis);
    return ret;
  };
  std::vector<Buffer> buffers = {
      out,
      device.CreateBufferFromScalar(sizeSortedAxis),
      device.CreateBufferFromScalar(outStrides[axis]),
      input,
      device.CreateBufferFromScalar(inputStrides[axis]),
  };
  auto outRestStrides = removeAxis(outStrides, axis);
  auto inputRestStrides = removeAxis(inputStrides, axis);
  bool contiguous = inputType == SortInputType::Contiguous;
  if (contiguous) {
    buffers.push_back(device.CreateBufferFromScalar(
        *std::min_element(outRestStrides.begin(), outRestStrides.end())));
    buffers.push_back(device.CreateBufferFromScalar(
        *std::min_element(inputRestStrides.begin(), inputRestStrides.end())));
  } else {
    auto inputRestShape = removeAxis(inputShape, axis);
    if (inputRestShape.empty()) {
      Buffer zero = device.CreateBufferFromScalar(
          0, DataType::U32, BufferUsage::Storage);
      buffers.push_back(zero);
      buffers.push_back(zero);
      buffers.push_back(zero);
    } else {
      buffers.push_back(device.CreateBufferFromVector(outRestStrides));
      buffers.push_back(device.CreateBufferFromVector(inputRestShape));
      buffers.push_back(device.CreateBufferFromVector(inputRestStrides));
    }
  }
  bool argsort = resultType == SortResultType::Indices;
  RunKernel(device,
            "sort_block",
            fmt::format("sort_{}_{}_{}",
                        WgslType(inputDataType),
                        argsort,
                        contiguous),
            [&]() {
              bool enableF16 = EnableF16(device, inputDataType);
              return Append(
                  ParseTemplate(wgsl_source_sort_block,
                                {
                                  {"enable_f16", enableF16},
                                  {"dtype", WgslType(inputDataType)},
                                  {"argsort", argsort},
                                  {"contiguous", contiguous},
                                }),
                  wgsl_source_utils,
                  ParseTemplate(wgsl_source_constants,
                                {
                                  {"enable_f16", enableF16},
                                  {"dtype", WgslType(inputDataType)},
                                }));
            },
            buffers,
            {1, NumElements(inputShape) / sizeSortedAxis, 1});
}

void UnaryOpContiguous(Device& device,
                       const char* name,
                       DataType outputDataType,
                       const Buffer& output,
                       DataType inputDataType,
                       const Buffer& input,
                       uint32_t inputNumElements) {
  const uint32_t workgroupSize = 64;  // TODO(zcbenz): make it dynamic
  uint32_t maxThreadsPerGridDim =
      device.GetLimits().maxComputeWorkgroupsPerDimension * workgroupSize;
  bool use2DGrid = inputNumElements > maxThreadsPerGridDim;
  bool inputIsIntegral = inputDataType == DataType::U32 ||
                         inputDataType == DataType::I32;
  RunKernel(device,
            fmt::format("unary_{}_{}", use2DGrid ? "v2" : "v", name),
            fmt::format("unary_{}_{}_{}",
                        name,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_unary_contiguous,
                      {
                        {"enable_f16", EnableF16(device,
                                                 outputDataType,
                                                 inputDataType)},
                        {"output_dtype", WgslType(outputDataType)},
                        {"input_dtype", WgslType(inputDataType)},
                        {"op", name},
                      }),
                  ParseTemplate(
                      wgsl_source_unary_ops,
                      {
                        {"input_is_bool", inputDataType == DataType::Bool},
                        {"input_is_floating", IsFloating(inputDataType)},
                        {"input_is_unsigned", IsUnsigned(inputDataType)},
                      }));
            },
            {output, input},
            GetWorkgroupsCountContiguous(inputNumElements,
                                         maxThreadsPerGridDim,
                                         workgroupSize));
}

void UnaryOpGeneral(Device& device,
                    const char* name,
                    DataType outputDataType,
                    const Buffer& output,
                    DataType inputDataType,
                    const Buffer& input,
                    const std::vector<uint32_t>& inputShapePre,
                    const std::vector<uint32_t>& inputStridesPre) {
  auto [inputShape, inputStrides] =
      CollapseContiguousDims(inputShapePre, inputStridesPre);
  if (inputShape.size() < 2)
    throw std::runtime_error("UnaryOpGeneral do not take contiguous inputs.");
  const uint32_t workgroupSize = 8;  // TODO(zcbenz): make it dynamic
  RunKernel(device,
            fmt::format("unary_g_{}", name),
            fmt::format("unary_g_{}_{}_{}",
                        name,
                        WgslType(outputDataType),
                        WgslType(inputDataType)),
            [&]() {
              return Append(
                  ParseTemplate(
                      wgsl_source_unary_general,
                      {
                        {"enable_f16", EnableF16(device,
                                                 outputDataType,
                                                 inputDataType)},
                        {"output_dtype", WgslType(outputDataType)},
                        {"input_dtype", WgslType(inputDataType)},
                        {"op", name},
                      }),
                  ParseTemplate(
                      wgsl_source_unary_ops,
                      {
                        {"input_is_bool", inputDataType == DataType::Bool},
                        {"input_is_floating", IsFloating(inputDataType)},
                        {"input_is_unsigned", IsUnsigned(inputDataType)},
                      }),
                  wgsl_source_utils);
            },
            {
              output,
              input,
              device.CreateBufferFromVector(inputShape),
              device.CreateBufferFromVector(inputStrides),
              device.CreateBufferFromStruct(GetDims(inputShape)),
            },
            GetWorkgroupsCountGeneral(inputShape, workgroupSize, 1));
}

}  // namespace betann
