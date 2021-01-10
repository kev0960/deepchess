#ifndef MOVE_H
#define MOVE_H

#include <cstdint>

namespace chess {

// Encapsulates the move that the piece can make.
class Move {
 public:
  constexpr Move(int from, int to) : from_to_((from << 6) | to) {}

  constexpr int From() const { return (from_to_ >> 6); }

  constexpr int To() const { return (from_to_ & 0b111111); }

 private:
  // [0 ~ 63][0 ~ 63]; The position on the board.
  uint16_t from_to_;
};

}  // namespace chess

#endif

