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
  constexpr Piece(PieceType type, PieceSide side)
      : info_(static_cast<int>(type) | (static_cast<int>(side) << 3)) {}

  explicit Piece(std::string_view piece);

  constexpr PieceType Type() const {
    // The last 3 bits encodes the type of the piece.
    return static_cast<PieceType>(info_ & 0b111);
  }

  constexpr PieceSide Side() const {
    return (info_ & 0b1000) == 0 ? WHITE : BLACK;
  }

  constexpr bool IsOpponent(PieceSide side) const {
    // Empty cell is NEITHER opponent or ally.
    if (Type() == EMPTY) {
      return false;
    }

    return Side() != side;
  }

  constexpr bool operator==(const Piece& piece) const {
    return info_ == piece.info_;
  }

  constexpr bool operator!=(const Piece& piece) const {
    return info_ != piece.info_;
  }

  void MarkPiece(uint64_t& board, int offset) const;

  char Print(char empty = ' ') const;

 private:
  // Data is encoded in the last 4 bits.
  // The first 4 bits must be 0. (This is because when we write to the data
  // array, we OR with the existing region).
  uint8_t info_;
};

}  // namespace chess

#endif
