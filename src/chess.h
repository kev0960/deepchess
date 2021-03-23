#ifndef CHESS_H
#define CHESS_H

#include "agent.h"

namespace chess {

enum GameResult { WHITE_WIN, DRAW, BLACK_WIN };

// Manages the entire chess game play.
class Chess {
 public:
  Chess(Config* config) : config_(config) {}

  // Play the game between the agent and human.
  GameResult PlayChessBetweenAgents(const Agent* white, const Agent* black);

  // Play the game between me and the trained agent.
  GameResult PlayChessWithHuman(const Agent* agent, PieceSide my_color);

 private:
  Config* config_ = nullptr;
};

}  // namespace chess

#endif
