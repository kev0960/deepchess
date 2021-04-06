#ifndef GAME_STATE_H
#define GAME_STATE_H

#include "board.h"
#include "util.h"

namespace chess {

struct GameStateSerialized {
  std::array<std::pair<Board, /*repetition_count=*/int>, 8> board_history;
  int num_history = 0;

  PieceSide who_is_moving;
  int total_move_count;
  int no_progress_count;

  // Castling availability of O-O and O-O-O.
  std::pair<bool, bool> p1_castle;
  std::pair<bool, bool> p2_castle;
};

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
  static GameState CreateGameStateForTesting(
      const Board& board, PieceSide who_is_moving = PieceSide::WHITE,
      Move last_move = Move(0, 0, 0, 0),
      CastlingAvail white_castle = CastlingAvail(),
      CastlingAvail black_castle = CastlingAvail());

  GameState(const GameState* prev_state, Move move);

  const Board& GetBoard() const { return current_board_; }
  const GameState* PrevState() const { return prev_state_; }
  const Move& LastMove() const { return last_move_; }
  PieceSide WhoIsMoving() const { return who_is_moving_; }

  // Returns (O-O, O-O-O)
  std::pair<bool, bool> CanWhiteCastle() const;
  std::pair<bool, bool> CanBlackCastle() const;

  int RepititionCount() const { return rep_count_; }
  int TotalMoveCount() const { return total_move_; }
  int NoProgressCount() const { return no_progress_count_; }

  // Get legal moves (the move that does not put King into check).
  std::vector<Move> GetLegalMoves() const;

  bool IsDraw() const;

  GameStateSerialized GetGameStateSerialized() const;

 private:
  // Should be only used by factory.
  GameState(const Board& board, PieceSide who_is_moving, Move last_move);

  std::vector<Move> ComputeLegalMoves() const;
  std::pair<bool, bool> ComputeCanWhiteCastle() const;
  std::pair<bool, bool> ComputeCanBlackCastle() const;

  Board current_board_;

  // The move that was made in the previous state to construct current state.
  Move last_move_;

  // The color that moved to reach this state.
  PieceSide who_is_moving_ = PieceSide::WHITE;

  CastlingAvail white_castle_, black_castle_;
  mutable LazyGet<std::pair<bool, bool>> can_white_castle_;
  mutable LazyGet<std::pair<bool, bool>> can_black_castle_;

  const GameState* prev_state_ = nullptr;

  // Repetition count. Includes the current state (so it always starts with 1).
  int rep_count_ = 1;

  // Total move count.
  int total_move_ = 0;

  // Increments when no capture or pawn move has been made. Resets to 0 if that
  // is done.
  int no_progress_count_ = 0;

  // Get legal moves. Once computed, it is cached here.
  mutable LazyGet<std::vector<Move>> legal_moves_;
};

}  // namespace chess

#endif
