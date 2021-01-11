#include "piece.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fmt/core.h"

namespace chess {
namespace {

using ::testing::Values;

class PieceParamTest
    : public testing::TestWithParam<std::tuple<PieceType, PieceSide, char>> {};

INSTANTIATE_TEST_SUITE_P(PieceParamTest, PieceParamTest,
                         Values(std::make_tuple(PAWN, BLACK, 'p'),
                                std::make_tuple(PAWN, WHITE, 'P'),
                                std::make_tuple(KNIGHT, BLACK, 'n'),
                                std::make_tuple(KNIGHT, WHITE, 'N'),
                                std::make_tuple(BISHOP, BLACK, 'b'),
                                std::make_tuple(BISHOP, WHITE, 'B'),
                                std::make_tuple(ROOK, BLACK, 'r'),
                                std::make_tuple(ROOK, WHITE, 'R'),
                                std::make_tuple(QUEEN, BLACK, 'q'),
                                std::make_tuple(QUEEN, WHITE, 'Q'),
                                std::make_tuple(KING, BLACK, 'k'),
                                std::make_tuple(KING, WHITE, 'K')));

TEST_P(PieceParamTest, Print) {
  auto [type, side, piece_char] = GetParam();

  Piece piece(type, side);
  EXPECT_EQ(piece.Print(), piece_char);
  EXPECT_EQ(piece.Type(), type);
  EXPECT_EQ(piece.Side(), side);

  Piece piece2(piece_char);
  EXPECT_EQ(piece.Type(), type);
  EXPECT_EQ(piece.Side(), side);
}

TEST(PieceTest, Mark) {
  Piece pawn(PAWN, BLACK);

  uint64_t board = 0;
  pawn.MarkPiece(board, 0);
  EXPECT_EQ(board, 0b1001);

  pawn.MarkPiece(board, 4);
  EXPECT_EQ(board, 0b10011001);

  pawn.MarkPiece(board, 12);
  EXPECT_EQ(board, 0b1001000010011001);

  Piece rook(ROOK, WHITE);
  rook.MarkPiece(board, 4);
  EXPECT_EQ(board, 0b1001000001001001);

  Piece queen(QUEEN, BLACK);
  queen.MarkPiece(board, 16);
  EXPECT_EQ(board, 0b11011001000001001001);

  queen.MarkPiece(board, 32);
  EXPECT_EQ(board, 0b110100000000000011011001000001001001);
}

TEST(PieceTest, Constructor) {
  Piece knight(0b11110010);

  EXPECT_EQ(knight.Type(), KNIGHT);
  EXPECT_EQ(knight.Side(), WHITE);

  uint64_t board = 0;
  knight.MarkPiece(board, 0);
  EXPECT_EQ(board, 0b0010);
}

}  // namespace
}  // namespace chess
