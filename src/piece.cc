#include "piece.h"

#include "fmt/core.h"
namespace chess {

Piece::Piece(PieceType type, PieceSide side) {
  info_ = 0;
  info_ |= (static_cast<int>(type) | (static_cast<int>(side) << 3));
}

Piece::Piece(std::string_view piece) {
  char ch = piece[0];

  info_ = 0;

  PieceType type;
  switch (std::tolower(ch)) {
    case 'p':
      type = PAWN;
      break;
    case 'n':
      type = KNIGHT;
      break;
    case 'b':
      type = BISHOP;
      break;
    case 'r':
      type = ROOK;
      break;
    case 'q':
      type = QUEEN;
      break;
    case 'k':
      type = KING;
      break;
    case ' ':
      type = EMPTY;
      break;
  }

  info_ |= (static_cast<int>(type) | (static_cast<int>(std::islower(ch)) << 3));
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
  const uint64_t clear_board = (-1) ^ (0b1111ll << offset);

  fmt::print("offset {} {:0>64b} {:0>64b}\n", offset, clear_board,
             uint64_t(0b1111ll << offset));

  // Scratch off the encoded region to 0.
  board &= clear_board;
  board |= (static_cast<uint64_t>(info_) << offset);
}

}  // namespace chess
