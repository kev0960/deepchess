#include "chess_nn.h"

namespace chess {

// Number of 8*8 planes in the input state.
constexpr int kStateSize = 119;
constexpr int kNumFilters = 50;

ChessNNImpl::ChessNNImpl(int num_layer) {
  conv_input_to_block_ = register_module(
      "conv_input_to_block",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(kStateSize, kNumFilters, 3)
                            .stride(1)
                            .padding(1)));

  layers_->push_back(conv_input_to_block_);

  for (int i = 0; i < num_layer; i++) {
    layers_->push_back(register_module("nn_block_" + std::to_string(i),
                                       ChessNNBlock(kNumFilters)));
  }

  register_module("chess_net", layers_);

  conv_policy_ = register_module(
      "conv_policy",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(kNumFilters, 73, {3, 3})
                            .stride(1)
                            .padding(1)));
  fc_policy_ =
      register_module("fc_policy", torch::nn::Linear(73 * 8 * 8, 73 * 8 * 8));

  conv_value_ = register_module(
      "conv_value",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(kNumFilters, 32, {3, 3})
                            .stride(1)
                            .padding(1)));

  fc_value_ = register_module("fc_value", torch::nn::Linear(32 * 8 * 8, 1));
}

torch::Tensor ChessNNImpl::GetPolicy(torch::Tensor state) {
  // If the state is [*, *, *], then make it as [1, *, *, *].
  if (state.sizes().size() == 3) {
    state = state.unsqueeze(0);
  }

  auto x = layers_->forward(state);

  // policy : N * 73 * 8 * 8
  auto policy = conv_policy_->forward(x);
  policy = torch::relu(policy);

  // policy : N * (73 * 8 * 8)
  policy = policy.flatten(1);
  policy = fc_policy_->forward(policy);

  return policy;
}

torch::Tensor ChessNNImpl::GetValue(torch::Tensor state) {
  // If the state is [*, *, *], then make it as [1, *, *, *].
  if (state.sizes().size() == 3) {
    state = state.unsqueeze(0);
  }

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
