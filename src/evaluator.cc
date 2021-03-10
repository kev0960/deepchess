#include "evaluator.h"

#include "nn/chess_nn.h"
#include "nn/nn_util.h"

namespace chess {

float Evaluator::Evalulate(const GameState& state) {
  torch::Tensor tensor = GameStateToTensor(state);

  // Convert board to the state.
  torch::Tensor value_tensor = chess_net_->GetValue(tensor);

  // Note that the returned value_tensor is 1 * 1.
  return value_tensor.data_ptr<float>()[0];
}

}  // namespace chess
