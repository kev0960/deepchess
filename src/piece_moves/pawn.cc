#include "pawn.h"

namespace chess {
namespace {

void AddPromotion(int row, int col, int next_row, int next_col,
                  std::vector<Move>* moves) {
  moves->push_back(Move(row, col, next_row, next_col, PROMOTE_QUEEN));
  moves->push_back(Move(row, col, next_row, next_col, PROMOTE_KNIGHT));
  moves->push_back(Move(row, col, next_row, next_col, PROMOTE_BISHOP));
  moves->push_back(Move(row, col, next_row, next_col, PROMOTE_ROOK));
}

}  // namespace

std::vector<Move> PawnMove::GetMoves(const Board& board, const Piece& piece,
                                     int row, int col) {
  std::vector<Move> moves;

  // For WHITE, it can only move UP. For Black, it can only move DOWN.
  int y_dir = piece.Side() == WHITE ? -1 : 1;

  int next_row = row + y_dir;

  // When the pawn reaches the end of the row, then it should be promoted.
  bool need_promo = false;
  if ((piece.Side() == WHITE && next_row == 0) ||
      (piece.Side() == BLACK && next_row == 7)) {
    need_promo = true;
  }

  // Pawn is already at the end.
  if (next_row < 0 || next_row >= 8) {
    return moves;
  }

  if (board.IsEmptyAt(next_row, col)) {
    if (need_promo) {
      AddPromotion(row, col, next_row, col, &moves);
    } else {
      moves.push_back(Move(row, col, next_row, col));
    }
  }

  // Can move diagonally when capturing the opponent piece.
  if (col + 1 < 8 &&
      board.PieceAt(next_row, col + 1).IsOpponent(piece.Side())) {
    if (need_promo) {
      AddPromotion(row, col, next_row, col + 1, &moves);
    } else {
      moves.push_back(Move(row, col, next_row, col + 1));
    }
  }

  if (col - 1 >= 0 &&
      board.PieceAt(next_row, col - 1).IsOpponent(piece.Side())) {
    if (need_promo) {
      AddPromotion(row, col, next_row, col - 1, &moves);
    } else {
      moves.push_back(Move(row, col, next_row, col - 1));
    }
  }

  // Can move two steps if the pawn is at the starting position.
  if (piece.Side() == WHITE && row == 6 && board.IsEmptyAt(4, col) &&
      board.IsEmptyAt(5, col)) {
    moves.push_back(Move(6, col, 4, col));
  }

  if (piece.Side() == BLACK && row == 1 && board.IsEmptyAt(3, col) &&
      board.IsEmptyAt(2, col)) {
    moves.push_back(Move(1, col, 3, col));
  }

  return moves;
}

}  // namespace chess
