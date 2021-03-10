#ifndef BITBOARD_H
#define BITBOARD_H

#include <array>

namespace chess {

// Following is the representation of the board in the 64bit integer.
//
// 8  0   1 ...  7
// 7  8   9 ... 15
//          ...
//          ...
// 1  56 57 ... 63
//    a  b      h
class Bitboard {
 public:
 private:
  std::array<uint64_t, 12> boards_;
};

}  // namespace chess

#endif
