#include <torch/torch.h>

namespace chess {

class SqueezeLayerImpl : torch::nn::Module {
 public:
  SqueezeLayerImpl(int num_channel, int reduction) {
    torch::nn::AvgPool2dOptions options({3, 4});

    avg_pool_ = register_module(
        "avgpool2d", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({8, 8})));
    fc1_ = register_module(
        "se-fc1", torch::nn::Linear(num_channel, num_channel / reduction));
    fc2_ = register_module(
        "se-fc2", torch::nn::Linear(num_channel / reduction, num_channel));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor original = x;

    // N * C * 8 * 8 --> N * C * 1 * 1
    x = avg_pool_->forward(x);

    // N * C * 1 * 1 --> N * C
    x = x.flatten(1);

    x = fc1_->forward(x);
    x = torch::relu(x);

    x = fc2_->forward(x);
    x = torch::sigmoid(x);

    // N * C --> N * C * 1 * 1
    x = x.unsqueeze(2);
    x = x.unsqueeze(3);

    x = x * original;
    return x;
  }

 private:
  torch::nn::AvgPool2d avg_pool_;
  torch::nn::Linear fc1_, fc2_;
};

TORCH_MODULE(SqueezeLayer);

class ChessNNBlock : torch::nn::Module {
 public:
  ChessNNBlock(int num_filters) {
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

    batch_norm1_ = register_module("batch_norm1", torch::nn::BatchNorm2d());
    batch_norm2_ = register_module("batch_norm2", torch::nn::BatchNorm2d());

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
  torch::nn::Linear fc1_, fc2_, fc3_;
  torch::nn::Conv2d conv1_, conv2_;
  torch::nn::BatchNorm2d batch_norm1_, batch_norm2_;
  SqueezeLayer se_layer_;
};

}  // namespace chess
