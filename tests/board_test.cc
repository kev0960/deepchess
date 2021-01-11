#include "board.h"

#include <fmt/core.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

TEST(BoardTest, SimpleBoard) {
  std::vector<PiecesOnBoard> pieces = {
      {"R", {"a1", "h1"}},
      {"K", {"b1", "g1"}},
      {"B", {"c1", "f1"}},
      {"Q", {"d1"}},
      {"K", {"e1"}},
      {"P", {"a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"}},
  };

  Board board(pieces);
  fmt::print("{}", board.PrintBoard());
}

}  // namespace
}  // namespace chess
