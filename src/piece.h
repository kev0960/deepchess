#ifndef PIECE_H
#define PIECE_H

#include <cstdint>
#include <string>

namespace chess {

enum PieceType { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum PieceSide { WHITE = 0, BLACK };

class Piece {
 public:
  constexpr Piece(uint8_t info) : info_(info & 0b00001111) {}
  Piece(PieceType type, PieceSide side);

  constexpr PieceType Type() const {
    // The last 3 bits encodes the type of the piece.
    return static_cast<PieceType>(info_ & 0b111);
  }

  constexpr PieceSide Side() const {
    return (info_ & 0b1000) == 0 ? WHITE : BLACK;
  }

  void MarkPiece(uint64_t& board, int offset) const;

  char Print() const;

 private:
  // Data is encoded in the last 4 bits.
  // The first 4 bits must be 0. (This is because when we write to the data
  // array, we OR with the existing region).
  uint8_t info_;
};

}  // namespace chess

#endif
