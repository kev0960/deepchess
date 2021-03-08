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
  static GameState CreateGameStateForTesting(const Board& board);

  GameState(const GameState* prev_state, Move move);

  const Board& GetBoard() const { return current_board_; }
  const GameState* PrevState() const { return prev_state_; }

  // Returns (O-O, O-O-O)
  std::pair<bool, bool> CanWhiteCastle() const;
  std::pair<bool, bool> CanBlackCastle() const;

  int RepititionCount() const { return rep_count_; }
  int TotalMoveCount() const { return total_move_; }
  int NoProgressCount() const { return no_progress_count_; }

 private:
  // Should be only used by factory.
  GameState(const Board& board);

  Board current_board_;

  CastlingAvail white_castle_, black_castle_;
  const GameState* prev_state_ = nullptr;

  // Repetition count. Includes the current state (so it always starts with 1).
  int rep_count_ = 1;

  // Total move count.
  int total_move_ = 0;

  // Increments when no capture or pawn move has been made. Resets to 0 if that
  // is done.
  int no_progress_count_ = 0;
};

}  // namespace chess

#endif
