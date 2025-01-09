#ifndef SRC_DEVICE_H_
#define SRC_DEVICE_H_

#include <dawn/webgpu_cpp.h>

namespace betann {

class Device {
 public:
  Device();
  ~Device();

 private:
  wgpu::Adapter adapter_;
  wgpu::Device device_;
};

}  // namespace betann

#endif  // SRC_DEVICE_H_
