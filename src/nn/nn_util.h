#ifndef NN_NN_UTIL_H
#define NN_NN_UTIL_H

#include <torch/torch.h>

#include <vector>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

// Convert board to tensor.
torch::Tensor GameStateToTensor(const GameState& current_state);
torch::Tensor GameStateSerializedToTensor(
    const GameStateSerialized& serialized);

// Convert (move, probability) pair to the policy tensor.
torch::Tensor MoveToTensor(std::vector<std::pair<Move, float>> move_and_prob);

int GetModelNumParams(ChessNN m);

// From the given policy vector, we have to mask out the impossible actions.
torch::Tensor NormalizePolicy(const GameState& game_state,
                              torch::Tensor policy);

}  // namespace chess

#endif
