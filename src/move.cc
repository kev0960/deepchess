#include "move.h"

namespace chess {
namespace {

std::string PromotionStr(Promotion p) {
  switch (p) {
    case NO_PROMOTE:
      return "";
    case PROMOTE_QUEEN:
      return "q";
    case PROMOTE_BISHOP:
      return "b";
    case PROMOTE_KNIGHT:
      return "n";
    case PROMOTE_ROOK:
      return "r";
  }

  return "";
}

std::string CoordToString(int row, int col) {
  std::string coord;

  coord.push_back(col + 'a');
  coord.push_back(7 - row + '1');

  return coord;
}

}  // namespace

Move Move::MoveFromString(std::string_view str_move) {
  if (str_move.size() < 4) {
    return Move(0, 0, 0, 0);
  }

  int from_col = str_move[0] - 'a';
  int from_row = 7 - (str_move[1] - '1');

  int to_col = str_move[2] - 'a';
  int to_row = 7 - (str_move[3] - '1');

  if (str_move.size() == 5) {
    Promotion promotion = NO_PROMOTE;
    switch (str_move[4]) {
      case 'q':
        promotion = PROMOTE_QUEEN;
        break;
      case 'n':
        promotion = PROMOTE_KNIGHT;
        break;
      case 'b':
        promotion = PROMOTE_BISHOP;
        break;
      case 'r':
        promotion = PROMOTE_ROOK;
        break;
    }

    return Move(from_row, from_col, to_row, to_col, promotion);
  }

  return Move(from_row, from_col, to_row, to_col);
}

std::string Move::FromStr() const {
  return CoordToString(From() / 8, From() % 8);
}

std::string Move::ToStr() const { return CoordToString(To() / 8, To() % 8); }

std::string Move::Str() const {
  return FromStr() + ToStr() + PromotionStr(promotion_);
}

}  // namespace chess
