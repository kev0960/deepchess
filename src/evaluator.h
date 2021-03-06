#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include "board.h"
#include "nn/chess_nn.h"

namespace chess {

class Evaluator {
 public:
  virtual float Evalulate(const Board& board, ChessNN* chess_net);

 private:
  ChessNN* chess_net_;
};

}  // namespace chess

#endif
