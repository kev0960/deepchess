#ifndef AGENT_H
#define AGENT_H

#include <torch/torch.h>

#include <vector>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

struct Experience {
  GameState* state = nullptr;
  torch::Tensor policy;

  // If the one who plays at this state wins, it should be 1. If lost, then -1.
  // Draw is 0.
  float result = 0;
};

class Agent {
 public:
  Agent();

  // Conduct the self play and gain experiences.
  void Run();
  void DoSelfPlay();

 private:
  std::vector<std::unique_ptr<Experience>> experiences_;
  std::vector<std::unique_ptr<GameState>> states_;

  ChessNN nn_;
};

}  // namespace chess

#endif
