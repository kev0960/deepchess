#ifndef MOVE_H
#define MOVE_H

#include <cstdint>
#include <string>

namespace chess {

// Encapsulates the move that the piece can make.
class Move {
 public:
  constexpr Move(int from, int to) : from_to_((from << 6) | to) {}
  constexpr Move(int row_from, int col_from, int row_to, int col_to)
      : from_to_(((row_from * 8 + col_from) << 6) | (row_to * 8 + col_to)) {}

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

  bool operator==(const Move& move) const { return from_to_ == move.from_to_; }
  bool operator!=(const Move& move) const { return from_to_ != move.from_to_; }

 private:
  // [0 ~ 63][0 ~ 63]; The position on the board.
  uint16_t from_to_;
};

}  // namespace chess

#endif

