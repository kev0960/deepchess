#include "game_state.h"

#include <fmt/core.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_utils.h"

namespace chess {
namespace {

using ::testing::IsSupersetOf;
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

TEST(GameStateTest, EnPassantWhite) {
  Board board = BoardFromNotation(R"(
r...k.r.
.p......
........
P.P.....
........
........
......P.
.R..K..R
)");

  GameStateBuilder builder = GameState::CreateGameStateForTesting(board, BLACK);
  builder.DoMove(Move(1, 1, 3, 1));

  auto moves = builder.GetStates().back()->GetLegalMoves();
  EXPECT_THAT(moves, IsSupersetOf({Move(3, 0, 2, 1), Move(3, 2, 2, 1)}));
}

TEST(GameStateTest, EnPassantBlack) {
  Board board = BoardFromNotation(R"(
r...k.r.
.p......
........
P.P.....
......p.
........
.......P
.R..K..R
)");

  GameStateBuilder builder = GameState::CreateGameStateForTesting(board, WHITE);
  builder.DoMove(Move(6, 7, 4, 7));

  auto moves = builder.GetStates().back()->GetLegalMoves();
  EXPECT_THAT(moves, IsSupersetOf({Move(4, 6, 5, 7)}));
}

TEST(GameStateTest, RepititionCountTest) {
  GameStateBuilder builder;

  builder
      .DoMove(Move(7, 1, 5, 2))  // Nb3
      .DoMove(Move(0, 1, 2, 2))  // Nc6
      /* White Knight goes back to original position. */
      .DoMove(Move(5, 2, 7, 1))
      /* Black Knight goes back to original position. */
      .DoMove(Move(2, 2, 0, 1));

  auto& states = builder.GetStates();
  EXPECT_EQ(states.back()->TotalMoveCount(), 4);
  EXPECT_EQ(states.back()->RepititionCount(), 2);
  EXPECT_EQ(states.back()->NoProgressCount(), 4);
}

TEST(GameStateTest, NoProgressCountTest) {
  GameStateBuilder builder;

  builder
      .DoMove(Move(7, 1, 5, 2))   // Nb3
      .DoMove(Move(0, 1, 2, 2));  // Nc6

  auto& states = builder.GetStates();
  EXPECT_EQ(states.back()->NoProgressCount(), 2);

  builder.DoMove(Move(6, 1, 4, 1));  // Makes a pawn move.
  EXPECT_EQ(states.back()->NoProgressCount(), 0);

  builder.DoMove(Move(2, 2, 4, 1));  // Knight captures pawn.
  EXPECT_EQ(states.back()->NoProgressCount(), 0);

  builder.DoMove(Move(7, 6, 5, 5));  // White knight moves.
  EXPECT_EQ(states.back()->NoProgressCount(), 1);
}

int DepthSearchMoves(const GameState& state, int depth) {
  int total = 0;

  auto moves = state.GetLegalMoves();
  if (depth == 0) {
    return moves.size();
  }

  for (auto m : moves) {
    GameState next_state(&state, m);
    total += DepthSearchMoves(next_state, depth - 1);
  }

  return total;
}

/*
TEST(GameStateTest, LegalMoveSanityTest) {
  Board board = BoardFromNotation(R"(
rnbq.k.r
pp.Pbppp
..p.....
........
..B.....
........
PPP.NnPP
RNBQK..R
)");

  GameState state = GameState::CreateGameStateForTesting(
      board, WHITE, Move(0, 0, 0, 0), {false, false, false},
      {true, false, false});

  auto moves = state.GetLegalMoves();
  int total = 0;
  for (auto m : moves) {
    GameState next_state(&state, m);
    int cur = DepthSearchMoves(next_state, 2);
    fmt::print("{} {} \n", m.Str(), cur);

    total += cur;
  }

  EXPECT_EQ(total, 2103487);
}
*/

}  // namespace
}  // namespace chess
