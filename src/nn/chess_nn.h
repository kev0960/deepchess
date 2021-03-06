#ifndef NN_CHESS_NN_H
#define NN_CHESS_NN_H

#include "chess_block.h"

namespace chess {

class ChessNN : public torch::nn::Module {
 public:
  ChessNN(int num_layer, int num_filter);

  virtual torch::Tensor GetPolicy(torch::Tensor state);
  virtual torch::Tensor GetValue(torch::Tensor state);

  torch::Tensor forward(torch::Tensor x) {
    x = layers_->forward(x);
    return x;
  }

 private:
  torch::nn::Sequential layers_;
  torch::nn::Conv2d conv_policy_{nullptr};
  torch::nn::Conv2d conv_value_{nullptr};
  torch::nn::Linear fc_policy_{nullptr};
  torch::nn::Linear fc_value_{nullptr};
};

}  // namespace chess

#endif
