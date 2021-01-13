#include "piece_move.h"

namespace chess {

void FetchMoveFromDelta(const Board& board, const Piece& piece,
                        std::vector<Move>& moves, int row, int col,
                        int delta_row, int delta_col, int num_step) {
  for (int step = 1; step <= num_step; step++) {
    const int new_row = row + delta_row * step;
    const int new_col = col + delta_col * step;

    if (new_row < 0 || new_row >= 8) {
      break;
    }

    if (new_col < 0 || new_col >= 8) {
      break;
    }

    if (board.IsEmptyAt(new_row, new_col)) {
      moves.push_back(Move(row, col, new_row, new_col));
    } else if (board.PieceAt(new_row, new_col).IsOpponent(piece.Side())) {
      // You can capture the opponent piece.
      moves.push_back(Move(row, col, new_row, new_col));
      break;
    } else {
      break;
    }
  }
}

}  // namespace chess
