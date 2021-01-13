#include "bishop.h"

#include "piece_moves/piece_move.h"

namespace chess {

std::vector<Move> BishopMove::GetMoves(const Board& board, const Piece& piece,
                                     int row, int col) {
  std::vector<Move> moves;

  FetchMoveFromDelta(board, piece, moves, row, col, 1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, 1, -1);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, 1);
  FetchMoveFromDelta(board, piece, moves, row, col, -1, -1);

  return moves;
}

}  // namespace chess
