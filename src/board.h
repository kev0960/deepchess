#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <cstdint>
#include <vector>

#include "move.h"
#include "piece.h"

namespace chess {

// Represents the pieces on the board.
// Each "position" on the board is represented by 4 bits.
// 1 bit --> Whether this is white (0) or black (1)
// 3 bits --> Indicate the type of the piece.
class Board {
 public:
  // Get list of possible moves.
  std::vector<Move> GetAvailableMoves() const;

  // Print the board.
  std::string PrintBoard() const;
  void PutPieceAt(int row, int col, Piece piece);

 private:
  Piece PieceAt(int row, int col) const;

  // Presence of pieces on the board.
  std::array<uint64_t, 4> board_;
};

}  // namespace chess

#endif
