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

std::string Move::FromStr() const {
  return CoordToString(From() / 8, From() % 8);
}

std::string Move::ToStr() const { return CoordToString(To() / 8, To() % 8); }

std::string Move::Str() const {
  return FromStr() + ToStr() + PromotionStr(promotion_);
}

}  // namespace chess
