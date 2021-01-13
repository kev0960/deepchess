#include "rook.h"

#include "piece_moves/piece_move.h"

namespace chess {

std::vector<Move> RookMove::GetMoves(const Board& board, const Piece& piece,
                                     int row, int col) {
  std::vector<Move> moves;

  // Check for the horizontal moves.
  FetchMoveFromDelta(board, piece, moves, row, col, 1, 0);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, 0);

  // Check for the vertical moves.
  FetchMoveFromDelta(board, piece, moves, row, col, 0, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, 0, -1);

  return moves;
}

}  // namespace chess
