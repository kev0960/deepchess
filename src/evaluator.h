#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

class Evaluator {
 public:
  virtual float Evalulate(const GameState& board, ChessNN* chess_net);

 private:
  ChessNN* chess_net_;
};

}  // namespace chess

#endif
