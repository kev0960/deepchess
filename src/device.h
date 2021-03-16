#ifndef DEVICE_H
#define DEVICE_H

#include "torch/torch.h"

namespace chess {

class DeviceManager {
 public:
  virtual torch::Device Device() {
    if (torch::cuda::is_available()) {
      return torch::Device(torch::kCUDA);
    } else {
      return torch::Device(torch::kCPU);
    }
  }
};

}  // namespace chess

#endif
