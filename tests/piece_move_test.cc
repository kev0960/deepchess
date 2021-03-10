#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "piece_moves/pawn.h"
#include "piece_moves/rook.h"

namespace chess {
namespace {

using ::testing::UnorderedElementsAre;

char PromotionToChar(Promotion p) {
  switch (p) {
    case PROMOTE_QUEEN:
      return 'Q';
    case PROMOTE_BISHOP:
      return 'B';
    case PROMOTE_KNIGHT:
      return 'N';
    case PROMOTE_ROOK:
      return 'R';
  }

  return 'P';
}

class PieceMoveTest : public testing::Test {
 protected:
  std::vector<std::string> ConvertMovesToChessNotation(
      const std::vector<Move> moves) {
    std::vector<std::string> notations;
    for (auto& m : moves) {
      notations.push_back(m.ToStr());
      if (m.GetPromotion() != NO_PROMOTE) {
        notations.back().push_back(PromotionToChar(m.GetPromotion()));
      }
    }

    return notations;
  }
};

TEST_F(PieceMoveTest, PawnSimpleForward) {
  std::vector<PiecesOnBoard> pieces = {{"P", {"a2"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("a3", "a4"));
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
                UnorderedElementsAre("a3", "b3", "a4"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("b3")),
                UnorderedElementsAre("a2", "b2"));
  }

  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"b2"}}, {"p", {"a3", "c3"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("b2")),
                UnorderedElementsAre("a3", "b3", "b4", "c3"));

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
                UnorderedElementsAre("b3", "b4"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("a3")),
                UnorderedElementsAre("a4"));

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("c3")),
                UnorderedElementsAre("c4"));
  }
}

TEST_F(PieceMoveTest, PawnPromotion) {
  {
    std::vector<PiecesOnBoard> pieces = {{"P", {"c7"}}, {"p", {"d8"}}};
    Board board(pieces);

    EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("c7")),
                UnorderedElementsAre("c8Q", "c8N", "c8B", "c8R", "d8Q", "d8N",
                                     "d8B", "d8R"));
  }
}

TEST_F(PieceMoveTest, RookSimple) {
  std::vector<PiecesOnBoard> pieces = {{"R", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("a4", "b4", "c4", "e4", "f4", "g4", "h4",
                                   "d1", "d2", "d3", "d5", "d6", "d7", "d8"));
}

TEST_F(PieceMoveTest, RookEdge) {
  std::vector<PiecesOnBoard> pieces = {{"R", {"a1"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("a2", "a3", "a4", "a5", "a6", "a7", "a8",
                                   "b1", "c1", "d1", "e1", "f1", "g1", "h1"));
}

TEST_F(PieceMoveTest, RookBlocked) {
  std::vector<PiecesOnBoard> pieces = {
      {"p", {"b4"}}, {"P", {"d7"}}, {"R", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("d4")),
              UnorderedElementsAre("b4", "c4", "e4", "f4", "g4", "h4", "d1",
                                   "d2", "d3", "d5", "d6"));
}

TEST_F(PieceMoveTest, BishopSimple) {
  std::vector<PiecesOnBoard> pieces = {{"B", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("c3", "b2", "a1", "e5", "f6", "g7", "h8",
                                   "c5", "b6", "a7", "e3", "f2", "g1"));
}

TEST_F(PieceMoveTest, BishopEdge) {
  std::vector<PiecesOnBoard> pieces = {{"B", {"a8"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("b7", "c6", "d5", "e4", "f3", "g2", "h1"));
}

TEST_F(PieceMoveTest, BishopBlocked) {
  std::vector<PiecesOnBoard> pieces = {
      {"p", {"b6"}}, {"P", {"g7"}}, {"B", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("d4")),
              UnorderedElementsAre("c3", "b2", "a1", "e5", "f6", "c5", "b6",
                                   "e3", "f2", "g1"));
}

TEST_F(PieceMoveTest, QueenSimple) {
  std::vector<PiecesOnBoard> pieces = {{"Q", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("c3", "b2", "a1", "e5", "f6", "g7", "h8",
                                   "c5", "b6", "a7", "e3", "f2", "g1", "a4",
                                   "b4", "c4", "e4", "f4", "g4", "h4", "d1",
                                   "d2", "d3", "d5", "d6", "d7", "d8"));
}

TEST_F(PieceMoveTest, QueenEdge) {
  std::vector<PiecesOnBoard> pieces = {{"Q", {"a8"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("b7", "c6", "d5", "e4", "f3", "g2", "h1",
                                   "a2", "a3", "a4", "a5", "a6", "a7", "a1",
                                   "b8", "c8", "d8", "e8", "f8", "g8", "h8"));
}

TEST_F(PieceMoveTest, QueenBlocked) {
  std::vector<PiecesOnBoard> pieces = {
      {"p", {"b6", "g4"}}, {"P", {"g7", "d7"}}, {"Q", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("d4")),
              UnorderedElementsAre("c3", "b2", "a1", "e5", "f6", "c5", "b6",
                                   "e3", "f2", "g1", "a4", "b4", "c4", "e4",
                                   "f4", "g4", "d1", "d2", "d3", "d5", "d6"));
}

TEST_F(PieceMoveTest, KnightSimple) {
  std::vector<PiecesOnBoard> pieces = {{"N", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(
      ConvertMovesToChessNotation(board.GetAvailableMoves()),
      UnorderedElementsAre("b5", "b3", "c2", "e2", "f3", "f5", "e6", "c6"));
}

TEST_F(PieceMoveTest, KnightEdge) {
  std::vector<PiecesOnBoard> pieces = {{"N", {"a8"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("c7", "b6"));
}

TEST_F(PieceMoveTest, KnightBlocked) {
  std::vector<PiecesOnBoard> pieces = {
      {"p", {"b3", "f5"}}, {"P", {"b5", "e2"}}, {"N", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("d4")),
              UnorderedElementsAre("b3", "c2", "f3", "f5", "e6", "c6"));
}

TEST_F(PieceMoveTest, KingSimple) {
  std::vector<PiecesOnBoard> pieces = {{"K", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(
      ConvertMovesToChessNotation(board.GetAvailableMoves()),
      UnorderedElementsAre("d3", "d5", "c3", "c4", "c5", "e3", "e4", "e5"));
}

TEST_F(PieceMoveTest, KingEdge) {
  std::vector<PiecesOnBoard> pieces = {{"K", {"a8"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetAvailableMoves()),
              UnorderedElementsAre("a7", "b7", "b8"));
}

TEST_F(PieceMoveTest, KingBlocked) {
  std::vector<PiecesOnBoard> pieces = {
      {"p", {"d3", "e4"}}, {"P", {"d5", "c4"}}, {"K", {"d4"}}};
  Board board(pieces);

  EXPECT_THAT(ConvertMovesToChessNotation(board.GetMoveOfPieceAt("d4")),
              UnorderedElementsAre("d3", "c3", "c5", "e3", "e4", "e5"));
}

}  // namespace
}  // namespace chess

