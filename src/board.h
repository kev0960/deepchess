#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <cstdint>
#include <vector>

#include "move.h"
#include "piece.h"

namespace chess {

struct PiecesOnBoard {
  std::string_view piece;  // Piece notation (based on piece.h)
  std::vector<std::string> pos;
};

// Represents the pieces on the board.
// Each "position" on the board is represented by 4 bits.
// 1 bit --> Whether this is white (0) or black (1)
// 3 bits --> Indicate the type of the piece.
class Board {
 public:
  // Create an empty board.
  Board();

  Board(const std::vector<PiecesOnBoard>& pieces);

  // Get list of possible moves.
  std::vector<Move> GetAvailableMoves() const;
  std::vector<Move> GetAvailableMoves(PieceSide who) const;

  // Get list of moves that I can do which does not make my king
  // checked.
  std::vector<Move> GetAvailableLegalMoves(PieceSide me) const;

  // Print the board.
  std::string PrintBoard(char empty = ' ') const;
  void PrettyPrintBoard() const;
  std::string PrintNumericBoard() const;

  // Put piece at specified coord.
  void PutPieceAt(int row, int col, Piece piece);
  void PutPieceAt(std::pair<int, int> coord, Piece piece);

  Piece PieceAt(int row, int col) const;
  Piece PieceAt(std::pair<int, int> coord) const;

  bool IsEmptyAt(int row, int col) const;

  bool IsCheck(PieceSide side) const;

  // Return true if there are only kings on the board.
  bool DrawByInsufficientMaterial() const;

  // Find the available moves of the piece at given location.
  std::vector<Move> GetMoveOfPieceAt(int row, int col) const;
  std::vector<Move> GetMoveOfPieceAt(std::string_view coord) const;

  // Do move specified in Move.
  Board DoMove(Move m) const;

  // Get every available move of certain side. This is useful when checking
  // castling conditions. Returns the binary 8 * 8 board (64 bits).
  uint64_t GetBinaryAvailableMoveOf(PieceSide side) const;

  uint64_t GetBinaryPositionOfAll() const;

  bool operator==(const Board& board) const;
  bool operator!=(const Board& board) const;

 private:
  // Presence of pieces on the board.
  std::array<uint64_t, 4> board_;
};

}  // namespace chess

#endif
