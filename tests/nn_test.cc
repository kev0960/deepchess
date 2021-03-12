#include <fmt/core.h>

#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nn/chess_nn.h"
#include "nn/nn_util.h"
#include "test_utils.h"

namespace chess {
namespace {

TEST(GameStateToTensorTest, GameStateToTensor) {
  const GameState state = GameState::CreateInitGameState();

  torch::Tensor tensor = GameStateToTensor(state);

  float white_pawn[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      1, 1, 1, 1, 1, 1, 1, 1,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  EXPECT_TRUE(tensor.index({0}).equal(torch::from_blob(white_pawn, {8, 8})));

  float black_pawn[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      1, 1, 1, 1, 1, 1, 1, 1,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  EXPECT_TRUE(tensor.index({6}).equal(torch::from_blob(black_pawn, {8, 8})));
}

TEST(GameStateToTensorTest, MultipleStates) {
  GameStateBuilder builder;

  builder
      .DoMove(Move(7, 1, 5, 2))   // Nb3
      .DoMove(Move(0, 1, 2, 2));  // Nc6

  auto& states = builder.GetStates();
  torch::Tensor tensor = GameStateToTensor(*states.back());

  // Let's check the position of knights.
  float white_knight_current[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 1, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 1, 0   // 1
  };

  float black_knight_current[] = {
      0, 0, 0, 0, 0, 0, 1, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 1, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  EXPECT_TRUE(
      tensor.index({1}).equal(torch::from_blob(white_knight_current, {8, 8})));
  EXPECT_TRUE(
      tensor.index({7}).equal(torch::from_blob(black_knight_current, {8, 8})));

  float white_knight_prev[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 1, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 1, 0   // 1
  };

  float black_knight_prev[] = {
      0, 1, 0, 0, 0, 0, 1, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  EXPECT_TRUE(
      tensor.index({15}).equal(torch::from_blob(white_knight_prev, {8, 8})));
  EXPECT_TRUE(
      tensor.index({21}).equal(torch::from_blob(black_knight_prev, {8, 8})));

  // Check repitition counter
  EXPECT_TRUE(tensor.index({118}).equal(torch::full({8, 8}, 2.0f)));

  // Check move counter.
  EXPECT_TRUE(tensor.index({113}).equal(torch::full({8, 8}, 2.0f)));
}

TEST(GameStateToTensorTest, Castling) {
  GameStateBuilder builder;

  builder
      .DoMove(Move(6, 4, 4, 4))  // e4
      .DoMove(Move(0, 1, 2, 2))  // Nb3
      .DoMove(Move(7, 6, 5, 5))  // Nc6
      .DoMove(Move(1, 0, 3, 0))  // a6
      .DoMove(Move(7, 5, 4, 2))  // Bc4
      .DoMove(
          Move(1, 1, 3, 1));  // b6 (Pawn move so no-progress count becomes 0)

  // Current state:
  // r bqkbnr
  //   pppppp
  //   n
  // pp
  //   B P
  //      N
  // PPPP PPP
  // RNBQK  R

  auto& states = builder.GetStates();
  torch::Tensor tensor = GameStateToTensor(*states.back());

  float white_bishop_current[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 1, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 1, 0, 0, 0, 0, 0   // 1
  };

  float white_knight_current[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 1, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 1, 0, 0, 0, 0, 0, 0   // 1
  };

  float white_pawn_current[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 1, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      1, 1, 1, 1, 0, 1, 1, 1,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  EXPECT_TRUE(
      tensor.index({0}).equal(torch::from_blob(white_pawn_current, {8, 8})));
  EXPECT_TRUE(
      tensor.index({1}).equal(torch::from_blob(white_knight_current, {8, 8})));
  EXPECT_TRUE(
      tensor.index({2}).equal(torch::from_blob(white_bishop_current, {8, 8})));

  // Check move counter.
  EXPECT_TRUE(tensor.index({113}).equal(torch::full({8, 8}, 6.0f)));

  // White king side castle (possible).
  EXPECT_TRUE(tensor.index({114}).equal(torch::full({8, 8}, 1.0f)));

  // Remaining castlings.
  EXPECT_TRUE(tensor.index({115}).equal(torch::full({8, 8}, 0.f)));
  EXPECT_TRUE(tensor.index({116}).equal(torch::full({8, 8}, 0.f)));
  EXPECT_TRUE(tensor.index({117}).equal(torch::full({8, 8}, 0.f)));
}

TEST(MoveToTensorTest, QueenMoveDiagonal) {
  torch::Tensor tensor = MoveToTensor({{Move(4, 4, 0, 0), 0.1},
                                       {Move(4, 4, 2, 6), 0.2},
                                       {Move(4, 4, 5, 5), 0.3},
                                       {Move(4, 4, 7, 1), 0.4}});
  float board[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  board[8 * 4 + 4] = 0.1;
  EXPECT_TRUE(tensor.index({7 * 7 + 3}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.2;
  EXPECT_TRUE(tensor.index({7 * 1 + 1}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.3;
  EXPECT_TRUE(tensor.index({7 * 3 + 0}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.4;
  EXPECT_TRUE(tensor.index({7 * 5 + 2}).equal(torch::from_blob(board, {8, 8})));
}

TEST(MoveToTensorTest, QueenMoveVerticalAndHorizontal) {
  torch::Tensor tensor = MoveToTensor({{Move(4, 4, 4, 0), 0.1},
                                       {Move(4, 4, 4, 7), 0.2},
                                       {Move(4, 4, 2, 4), 0.3},
                                       {Move(4, 4, 5, 4), 0.4}});
  float board[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  board[8 * 4 + 4] = 0.1;
  EXPECT_TRUE(tensor.index({7 * 6 + 3}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.2;
  EXPECT_TRUE(tensor.index({7 * 2 + 2}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.3;
  EXPECT_TRUE(tensor.index({7 * 0 + 1}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.4;
  EXPECT_TRUE(tensor.index({7 * 4 + 0}).equal(torch::from_blob(board, {8, 8})));
}

TEST(MoveToTensorTest, KnightMove) {
  // NOTE: probability should sum to 1 but let's just allow it for test's sake.
  torch::Tensor tensor = MoveToTensor({{Move(4, 4, 2, 5), 0.1},
                                       {Move(4, 4, 3, 6), 0.2},
                                       {Move(4, 4, 5, 6), 0.3},
                                       {Move(4, 4, 6, 5), 0.4},
                                       {Move(4, 4, 6, 3), 0.5},
                                       {Move(4, 4, 5, 2), 0.6},
                                       {Move(4, 4, 3, 2), 0.7},
                                       {Move(4, 4, 2, 3), 0.8}});
  float board[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  board[8 * 4 + 4] = 0.1;
  EXPECT_TRUE(tensor.index({7 * 8 + 0}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.2;
  EXPECT_TRUE(tensor.index({7 * 8 + 2}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.3;
  EXPECT_TRUE(tensor.index({7 * 8 + 4}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.4;
  EXPECT_TRUE(tensor.index({7 * 8 + 6}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.5;
  EXPECT_TRUE(tensor.index({7 * 8 + 7}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.6;
  EXPECT_TRUE(tensor.index({7 * 8 + 5}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.7;
  EXPECT_TRUE(tensor.index({7 * 8 + 3}).equal(torch::from_blob(board, {8, 8})));

  board[8 * 4 + 4] = 0.8;
  EXPECT_TRUE(tensor.index({7 * 8 + 1}).equal(torch::from_blob(board, {8, 8})));
}

TEST(MoveToTensorTest, PawnPromotion) {
  torch::Tensor tensor = MoveToTensor({{Move(6, 4, 7, 3, PROMOTE_QUEEN), 0.1},
                                       {Move(6, 4, 7, 4, PROMOTE_QUEEN), 0.2},
                                       {Move(6, 4, 7, 5, PROMOTE_QUEEN), 0.3},
                                       {Move(6, 4, 7, 3, PROMOTE_BISHOP), 0.4},
                                       {Move(6, 4, 7, 4, PROMOTE_BISHOP), 0.5},
                                       {Move(6, 4, 7, 5, PROMOTE_BISHOP), 0.6},
                                       {Move(6, 4, 7, 3, PROMOTE_KNIGHT), 0.7},
                                       {Move(6, 4, 7, 4, PROMOTE_KNIGHT), 0.8},
                                       {Move(6, 4, 7, 5, PROMOTE_KNIGHT), 0.9},
                                       {Move(6, 4, 7, 3, PROMOTE_ROOK), 1},
                                       {Move(6, 4, 7, 4, PROMOTE_ROOK), 1.1},
                                       {Move(6, 4, 7, 5, PROMOTE_ROOK), 1.2}});

  float board[] = {
      0, 0, 0, 0, 0, 0, 0, 0,  // 8
      0, 0, 0, 0, 0, 0, 0, 0,  // 7
      0, 0, 0, 0, 0, 0, 0, 0,  // 6
      0, 0, 0, 0, 0, 0, 0, 0,  // 5
      0, 0, 0, 0, 0, 0, 0, 0,  // 4
      0, 0, 0, 0, 0, 0, 0, 0,  // 3
      0, 0, 0, 0, 0, 0, 0, 0,  // 2
      0, 0, 0, 0, 0, 0, 0, 0   // 1
  };

  int indices[] = {7 * 5,  7 * 4,  7 * 3,  67 + 0, 67 + 1, 67 + 2,
                   64 + 0, 64 + 1, 64 + 2, 70 + 0, 70 + 1, 70 + 2};

  for (int i = 1; i <= 12; i++) {
    board[8 * 6 + 4] = 0.1 * i;
    EXPECT_TRUE(
        tensor.index({indices[i - 1]}).equal(torch::from_blob(board, {8, 8})));
  }
}

TEST(ChessNNTest, ChessNN) {
  ChessNN model(152, 119);

  fmt::print("Total : {} {}MB\n", GetModelNumParams(model),
             GetModelNumParams(model) * 4 / 1024.f / 1024.f);
}

}  // namespace
}  // namespace chess

