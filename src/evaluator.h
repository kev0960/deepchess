#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include "config.h"
#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

class Evaluator {
 public:
  Evaluator(ChessNN chess_net, const Config* config)
      : chess_net_(chess_net), config_(config) {}

  virtual float Evalulate(const GameState& board);
  virtual std::vector<float> EvalulateBatch(
      std::vector<const GameState*> boards);

 private:
  ChessNN chess_net_;
  const Config* config_;
};

}  // namespace chess

#endif
