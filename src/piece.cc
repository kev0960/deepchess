#include "piece.h"

namespace chess {

Piece::Piece(PieceType type, PieceSide side) {
  info_ = 0;
  info_ |= (static_cast<int>(type) | (static_cast<int>(side) << 3));
}

char Piece::Print() const {
  switch (Type()) {
    case EMPTY:
      return ' ';
    case PAWN:
      return Side() ? 'p' : 'P';
    case KNIGHT:
      return Side() ? 'n' : 'N';
    case BISHOP:
      return Side() ? 'b' : 'B';
    case ROOK:
      return Side() ? 'r' : 'R';
    case QUEEN:
      return Side() ? 'q' : 'Q';
    case KING:
      return Side() ? 'k' : 'K';
    default:
      return ' ';
  }
}

void Piece::MarkPiece(uint64_t& board, int offset) const {
  // 11111 .... 1 0000 111 ... 111111
  //                   <-- offset -->
  const uint64_t clear_board = (-1) ^ (0b1111 << offset);

  // Scratch off the encoded region to 0.
  board &= clear_board;
  board |= (info_ << offset);
}

}  // namespace chess
