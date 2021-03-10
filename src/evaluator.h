#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

class Evaluator {
 public:
  Evaluator(ChessNN* chess_net) : chess_net_(chess_net) {}

  virtual float Evalulate(const GameState& board);

 private:
  ChessNN* chess_net_;
};

}  // namespace chess

#endif
