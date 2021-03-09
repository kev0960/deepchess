#include "board.h"

#include <fmt/core.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_utils.h"

namespace chess {
namespace {

using ::testing::UnorderedElementsAreArray;

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

TEST(BoardTest, TestCheck) {
  const Board b = BoardFromNotation(R"(
....k...
....q...
........
........
........
........
....B...
....K...
)");

  const std::vector<Move> moves = b.GetAvailableLegalMoves(PieceSide::WHITE);
  for (auto m : moves) {
    fmt::print("m : {}  \n", m.Str());
  }

  EXPECT_THAT(moves,
              UnorderedElementsAreArray({Move(7, 4, 7, 3), Move(7, 4, 7, 5),
                                         Move(7, 4, 6, 3), Move(7, 4, 6, 5)}));
}

TEST(BoardTest, TestCheck2) {
  const Board b = BoardFromNotation(R"(
....k...
....q...
........
........
........
..p...p.
....B...
....K...
)");

  const std::vector<Move> moves = b.GetAvailableLegalMoves(PieceSide::WHITE);
  EXPECT_THAT(moves,
              UnorderedElementsAreArray({Move(7, 4, 7, 3), Move(7, 4, 7, 5)}));
}

TEST(BoardTest, TestCheckMate) {
  const Board b = BoardFromNotation(R"(
....k...
........
......r.
........
....b...
........
.......P
.......K
)");

  const std::vector<Move> moves = b.GetAvailableLegalMoves(PieceSide::WHITE);
  EXPECT_EQ(moves.size(), 0);
}

}  // namespace
}  // namespace chess
