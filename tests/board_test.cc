#include "board.h"

#include <fmt/core.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chess {
namespace {

TEST(BoardTest, SimpleBoard) {
  std::vector<PiecesOnBoard> pieces = {
      {"R", {"a1", "h1"}},
      {"N", {"b1", "g1"}},
      {"B", {"c1", "f1"}},
      {"Q", {"d1"}},
      {"K", {"e1"}},
      {"P", {"a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"}},
      {"r", {"a8", "h8"}},
      {"n", {"b8", "g8"}},
      {"b", {"c8", "f8"}},
      {"q", {"d8"}},
      {"k", {"e8"}},
      {"p", {"a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"}},
  };

  Board board(pieces);

  std::string chess_board = R"(rnbqkbnr
pppppppp
        
        
        
        
PPPPPPPP
RNBQKBNR
)";

  EXPECT_EQ(board.PrintBoard(), chess_board);
}

}  // namespace
}  // namespace chess
