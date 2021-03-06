#include <torch/torch.h>

#include "se_layer.h"

namespace chess {

class ChessNNBlockImpl : public torch::nn::Module {
 public:
  ChessNNBlockImpl(int num_filters) {
    fc1_ = register_module("fc1", torch::nn::Linear(100, 50));
    fc2_ = register_module("fc2", torch::nn::Linear(50, 20));
    fc3_ = register_module("fc3", torch::nn::Linear(20, 10));

    conv1_ = register_module(
        "conv1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(117, num_filters, {3, 3}).stride(1)));
    conv2_ = register_module(
        "conv2",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(117, num_filters, {3, 3}).stride(1)));

    batch_norm1_ = register_module("batch_norm1", torch::nn::BatchNorm2d(117));
    batch_norm2_ = register_module("batch_norm2", torch::nn::BatchNorm2d(117));

    se_layer_ = register_module("se_layer", SqueezeLayer(10, 5));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto original = x;

    x = conv1_->forward(x);
    x = batch_norm1_->forward(x);
    x = torch::relu(x);

    x = conv2_->forward(x);
    x = batch_norm2_->forward(x);
    x = torch::relu(x);
    x = se_layer_->forward(x);

    // Residual connection.
    x = x + original;

    x = torch::relu(x);
    return x;
  }

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
  torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
  torch::nn::BatchNorm2d batch_norm1_{nullptr}, batch_norm2_{nullptr};

  SqueezeLayer se_layer_{nullptr};
};

TORCH_MODULE(ChessNNBlock);

}  // namespace chess
