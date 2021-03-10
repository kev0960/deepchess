#include "chess_nn.h"

namespace chess {

ChessNN::ChessNN(int num_layer, int num_filter) {
  for (int i = 0; i < num_layer; i++) {
    layers_->push_back(register_module("nn_block_" + std::to_string(i),
                                       ChessNNBlock(num_filter)));
  }

  register_module("chess_net", layers_);

  conv_policy_ = register_module(
      "conv_policy",
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(num_filter, 73, {3, 3}).stride(1)));
  fc_policy_ =
      register_module("fc_policy", torch::nn::Linear(73 * 8 * 8, 73 * 8 * 8));

  conv_value_ = register_module(
      "conv_value",
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(num_filter, 32, {3, 3}).stride(1)));

  fc_value_ = register_module("fc_value", torch::nn::Linear(32 * 8 * 8, 1));
}

torch::Tensor ChessNN::GetPolicy(torch::Tensor state) {
  auto x = layers_->forward(state);

  // policy : N * 73 * 8 * 8
  auto policy = conv_policy_->forward(x);
  policy = torch::relu(policy);

  // policy : N * (73 * 8 * 8)
  policy = policy.flatten(1);
  policy = fc_policy_->forward(policy);

  return policy;
}

torch::Tensor ChessNN::GetValue(torch::Tensor state) {
  auto x = layers_->forward(state);

  // value : N * 32 * 8 * 8
  auto value = conv_value_->forward(x);
  value = torch::relu(value);

  // value : N * (32 * 8 * 8)
  value = value.flatten(1);

  // value : N * 1
  value = fc_value_->forward(value);

  return value;
}

}  // namespace chess
