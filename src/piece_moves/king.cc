#include "king.h"

#include "piece_moves/piece_move.h"

namespace chess {

std::vector<Move> KingMove::GetMoves(const Board& board, const Piece& piece,
                                     int row, int col) {
  std::vector<Move> moves;

  FetchMoveFromDelta(board, piece, moves, row, col, 1, 0, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, 0, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, 0, 1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, 0, -1, 1);

  FetchMoveFromDelta(board, piece, moves, row, col, 1, 1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, 1, -1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, 1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, -1, 1);

  return moves;
}

}  // namespace chess
