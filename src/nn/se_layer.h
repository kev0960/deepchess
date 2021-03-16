#include <torch/torch.h>
#include <fmt/core.h>

namespace chess {

class SqueezeLayerImpl : public torch::nn::Module {
 public:
  SqueezeLayerImpl(int num_channel, int reduction) {
    assert(num_channel / reduction * reduction == num_channel);

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
  torch::nn::AvgPool2d avg_pool_{nullptr};
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

TORCH_MODULE(SqueezeLayer);

}  // namespace chess
