#ifndef AGENT_H
#define AGENT_H

#include <torch/torch.h>

#include <vector>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

struct Experience {
  torch::Tensor state_;
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
