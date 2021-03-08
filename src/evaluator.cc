#include "evaluator.h"

#include "nn/chess_nn.h"

namespace chess {

float Evaluator::Evalulate(const GameState& state, ChessNN* chess_net) {
  // Convert board to the state.
  (void)state;
  (void)chess_net;

  return 0;
}

}  // namespace chess
