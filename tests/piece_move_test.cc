#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "piece_moves/pawn.h"

namespace chess {
namespace {

using ::testing::UnorderedElementsAre;

class PieceMoveTest : public testing::Test {
 protected:
  std::vector<std::string> ConvertMovesToChessNotation(
      const std::vector<Move> moves) {
    std::vector<std::string> notations;
    for (auto& m : moves) {
      notations.push_back(m.ToStr());
    }

    return notations;
  }
};

TEST_F(PieceMoveTest, PawnSimpleForward) {
  std::vector<PiecesOnBoard> pieces = {{"P", {"a2"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("a3"));
}

TEST_F(PieceMoveTest, PawnAtEdge) {
  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"a8"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
                UnorderedElementsAre());
  }

  {
    std::vector<PiecesOnBoard> pieces = {{"p", {"a1"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
                UnorderedElementsAre());
  }
}

TEST_F(PieceMoveTest, PawnBlockedForward) {
  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"a2", "a3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a2")),
                UnorderedElementsAre());

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a3")),
                UnorderedElementsAre("a4"));
  }

  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"a2"}}, {"p", {"a3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a2")),
                UnorderedElementsAre());

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a3")),
                UnorderedElementsAre());
  }
}

TEST_F(PieceMoveTest, PawnCapture) {
  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"a2"}}, {"p", {"b3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a2")),
                UnorderedElementsAre("a3", "b3"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("b3")),
                UnorderedElementsAre("a2", "b2"));
  }

  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"b2"}}, {"p", {"a3", "c3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("b2")),
                UnorderedElementsAre("a3", "b3", "c3"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a3")),
                UnorderedElementsAre("b2", "a2"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("c3")),
                UnorderedElementsAre("b2", "c2"));
  }

  // Do not capture ally piece.
  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"b2", "a3", "c3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("b2")),
                UnorderedElementsAre("b3"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a3")),
                UnorderedElementsAre("a4"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("c3")),
                UnorderedElementsAre("c4"));
  }
}

}  // namespace
}  // namespace chess

