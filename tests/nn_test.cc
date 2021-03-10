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

TEST(ChessNNTest, ChessNN) {
  ChessNN model(152, 119);

  fmt::print("Total : {} {}MB\n", GetModelNumParams(model),
             GetModelNumParams(model) * 4 / 1024.f / 1024.f);
}

}  // namespace
}  // namespace chess

