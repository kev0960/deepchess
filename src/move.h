#ifndef MOVE_H
#define MOVE_H

#include <cstdint>
#include <string>

namespace chess {

enum Promotion {
  NO_PROMOTE,
  PROMOTE_QUEEN,
  PROMOTE_KNIGHT,
  PROMOTE_BISHOP,
  PROMOTE_ROOK,
};

// Encapsulates the move that the piece can make.
class Move {
 public:
  constexpr Move(int from, int to, Promotion promotion = NO_PROMOTE)
      : from_to_((from << 6) | to), promotion_(promotion) {}
  constexpr Move(int row_from, int col_from, int row_to, int col_to,
                 Promotion promotion = NO_PROMOTE)
      : from_to_(((row_from * 8 + col_from) << 6) | (row_to * 8 + col_to)),
        promotion_(promotion) {}

  constexpr int From() const { return static_cast<char>(from_to_ >> 6); }
  constexpr int To() const { return static_cast<char>(from_to_ & 0b111111); }

  constexpr std::pair<int, int> FromCoord() const {
    return std::make_pair(From() / 8, From() % 8);
  }

  constexpr std::pair<int, int> ToCoord() const {
    return std::make_pair(To() / 8, To() % 8);
  }

  std::string FromStr() const;
  std::string ToStr() const;
  std::string Str() const;
  Promotion GetPromotion() const { return promotion_; }

  bool operator==(const Move& move) const {
    return (from_to_ == move.from_to_) && (promotion_ == move.promotion_);
  }

  bool operator!=(const Move& move) const { return !(operator==(move)); }

 private:
  // [0 ~ 63][0 ~ 63]; The position on the board.
  uint16_t from_to_;

  // 0: Queen, 1: Knight, 2: Bishop, 3: Rook
  Promotion promotion_ = NO_PROMOTE;
};

}  // namespace chess

#endif

