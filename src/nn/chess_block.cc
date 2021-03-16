#include "chess_block.h"

namespace chess {

ChessNNBlockImpl::ChessNNBlockImpl(int num_filters) {
  fc1_ = register_module("fc1", torch::nn::Linear(100, 50));
  fc2_ = register_module("fc2", torch::nn::Linear(50, 20));
  fc3_ = register_module("fc3", torch::nn::Linear(20, 10));

  conv1_ = register_module(
      "conv1", torch::nn::Conv2d(
                   torch::nn::Conv2dOptions(num_filters, num_filters, {3, 3})
                       .stride(1)
                       .padding(1)));
  conv2_ = register_module(
      "conv2", torch::nn::Conv2d(
                   torch::nn::Conv2dOptions(num_filters, num_filters, {3, 3})
                       .stride(1)
                       .padding(1)));

  batch_norm1_ =
      register_module("batch_norm1", torch::nn::BatchNorm2d(num_filters));
  batch_norm2_ =
      register_module("batch_norm2", torch::nn::BatchNorm2d(num_filters));

  se_layer_ = register_module("se_layer", SqueezeLayer(num_filters, 10));
}

torch::Tensor ChessNNBlockImpl::forward(torch::Tensor x) {
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

}  // namespace chess
