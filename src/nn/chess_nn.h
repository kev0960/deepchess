#ifndef NN_CHESS_NN_H
#define NN_CHESS_NN_H

#include "chess_block.h"
#include "device.h"

namespace chess {

class ChessNNImpl : public torch::nn::Module {
 public:
  ChessNNImpl(int num_layer);

  virtual torch::Tensor GetPolicy(torch::Tensor state);
  virtual torch::Tensor GetValue(torch::Tensor state);

 private:
  torch::nn::Sequential layers_;
  torch::nn::Conv2d conv_input_to_block_{nullptr};
  torch::nn::Conv2d conv_policy_{nullptr};
  torch::nn::Conv2d conv_value_{nullptr};
  torch::nn::Linear fc_policy_{nullptr};
  torch::nn::Linear fc_value_{nullptr};
};

TORCH_MODULE(ChessNN);

}  // namespace chess

#endif
