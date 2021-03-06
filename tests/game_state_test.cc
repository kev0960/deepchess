#include "game_state.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_utils.h"

namespace chess {
namespace {

using ::testing::Pair;

TEST(GameStateTest, KingChecked) {
  Board board = BoardFromNotation(R"(
r...k..r
........
........
.Q......
........
........
........
....K...
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(false, false));
}

TEST(GameStateTest, AttackBetween) {
  Board board = BoardFromNotation(R"(
....k...
......q.
........
........
........
........
........
R...K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(false, true));
}

TEST(GameStateTest, AttackBetweenBlack) {
  Board board = BoardFromNotation(R"(
r...k..r
........
........
........
........
......Q.
........
R...K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(true, true));
}

TEST(GameStateTest, AttackBetween2) {
  Board board = BoardFromNotation(R"(
....k...
........
........
........
......q.
........
........
R...K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(false, false));
}

TEST(GameStateTest, AttackBetween2Black) {
  Board board = BoardFromNotation(R"(
r...k..r
.Q......
........
........
........
........
........
R...K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(true, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(true, true));
}

TEST(GameStateTest, ObstacleWhite) {
  Board board = BoardFromNotation(R"(
r...k..r
........
........
........
........
........
........
R.B.KB.R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(true, true));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(false, false));
}

TEST(GameStateTest, ObstacleBlack) {
  Board board = BoardFromNotation(R"(
rn..kb.r
........
........
........
........
........
........
R...K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, false));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(true, true));
}

TEST(GameStateTest, RookMoved) {
  Board board = BoardFromNotation(R"(
r...k.r.
.p......
........
........
........
........
......P.
.R..K..R
)");

  GameState state = GameState::CreateGameStateForTesting(board);

  EXPECT_THAT(state.CanBlackCastle(), Pair(false, true));
  EXPECT_THAT(state.CanWhiteCastle(), Pair(true, false));
}

}  // namespace
}  // namespace chess
