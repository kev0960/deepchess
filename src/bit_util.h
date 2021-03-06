#ifndef BIT_UTIL_H
#define BIT_UTIL_H

namespace chess {

template <typename Int>
Int OnBitAt(Int bits, int index) {
  Int bit = (1 << index);
  return bits | bit;
}

template <typename Int>
Int OffBitAt(Int bits, int index) {
  Int bit = -1; // 0b11111...111
  bit = bit ^ (1 << index); // 0b11110111...11

  return bits &  bit;
}

}  // namespace chess

#endif
