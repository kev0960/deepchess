#include "board.h"

namespace chess {

Piece Board::PieceAt(int row, int col) const {
  const int index = 4 * (8 * row + col) / 64;
  const int offset_in_board_elem = 4 * (8 * row + col) % 64;

  // Extract 4 bit information about the piece.
  uint8_t info = (board_[index] & (0b1111 << offset_in_board_elem)) >>
                 offset_in_board_elem;

  return Piece(info);
}

std::vector<Move> Board::GetAvailableMoves() const {
  std::vector<Move> moves;

  return moves;
}

std::string Board::PrintBoard() const {
  std::string board;
  board.reserve(64 + 8);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      board.push_back(PieceAt(i, j).Print());
    }
    board.push_back('\n');
  }

  return board;
}

void Board::PutPieceAt(int row, int col, Piece piece) {
  const int index = 4 * (8 * row + col) / 64;
  const int offset_in_board_elem = 4 * (8 * row + col) % 64;

  piece.MarkPiece(board_[index], offset_in_board_elem);
}

}  // namespace chess
