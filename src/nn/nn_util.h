#ifndef NN_NN_UTIL_H
#define NN_NN_UTIL_H

#include <torch/torch.h>

#include <vector>

#include "game_state.h"

namespace chess {

// Convert board to tensor.
torch::Tensor GameStateToTensor(const GameState& current_state);

int GetModelNumParams(const torch::nn::Module& m);

}  // namespace chess

#endif
