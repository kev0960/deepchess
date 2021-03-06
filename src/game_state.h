#ifndef GAME_STATE_H
#define GAME_STATE_H

#include "board.h"

namespace chess {

struct CastlingAvail {
  bool king_side_rook_moved = false;
  bool queen_side_rook_moved = false;
  bool king_moved = false;
};

// Current game state. This captures current board, castling availability,
// enpassant and so on.
class GameState {
 public:
  // Create the init game state.
  static GameState CreateInitGameState();

  GameState(const GameState* prev_state, Move move);

  const Board& GetBoard() const { return current_board_; }

  // Returns (O-O, O-O-O)
  std::pair<bool, bool> CanWhiteCastle() const;
  std::pair<bool, bool> CanBlackCastle() const;

 private:
  // Should be only used by factory.
  GameState(const Board& board);

  Board current_board_;

  CastlingAvail white_castle_, black_castle_;
  const GameState* prev_state_ = nullptr;
};

}  // namespace chess

#endif
