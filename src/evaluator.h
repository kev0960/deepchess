#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

class Evaluator {
 public:
  Evaluator(ChessNN chess_net, DeviceManager* device_manager)
      : chess_net_(chess_net), device_manager_(device_manager) {}

  virtual float Evalulate(const GameState& board);
  virtual std::vector<float> EvalulateBatch(std::vector<const GameState*> boards);

 private:
  ChessNN chess_net_;
  DeviceManager* device_manager_;
};

}  // namespace chess

#endif
