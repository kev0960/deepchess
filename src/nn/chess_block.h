#include <torch/torch.h>

#include "se_layer.h"

namespace chess {

class ChessNNBlockImpl : public torch::nn::Module {
 public:
  ChessNNBlockImpl(int num_filters);

  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
  torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
  torch::nn::BatchNorm2d batch_norm1_{nullptr}, batch_norm2_{nullptr};

  SqueezeLayer se_layer_{nullptr};
};

TORCH_MODULE(ChessNNBlock);

}  // namespace chess
