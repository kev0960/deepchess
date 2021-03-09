#ifndef BITBOARD_H
#define BITBOARD_H

#include <array>

namespace chess {

// Following is the representation of the board in the 64bit integer.
//
// 8  56 57 ... 63
// 7  48 49 ... 55
//        ...
//        ...
// 1  0  1      7
//    a  b      h
class Bitboard {
 public:
 private:
  std::array<uint64_t, 12> boards_;
};

}  // namespace chess

#endif
