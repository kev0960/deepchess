#include "bit_util.h"

namespace chess {

std::string StrBinaryBoard(uint64_t binary_board) {
  std::string s;
  s.reserve(64 + 8);

  for (int i = 0; i < 64; i++) {
    if (i > 0 && i % 8 == 0) {
      s.push_back('\n');
    }

    s.push_back(GetBitAt(binary_board, i) ? '1' : '0');
  }

  return s;
}

}  // namespace chess
