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

  std::string FromStr() const; 
  std::string ToStr() const;

 private:
  // [0 ~ 63][0 ~ 63]; The position on the board.
  uint16_t from_to_;
};

}  // namespace chess

#endif

