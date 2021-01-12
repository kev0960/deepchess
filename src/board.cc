#include "board.h"

#include <fmt/core.h>

namespace chess {
namespace {

std::pair<int, int> ChessNotationToCoord(std::string_view notation) {
  if (notation.size() != 2) {
    return std::make_pair(0, 0);
  }

  int row = 7 - (notation[1] - '1');
  int col = notation[0] - 'a';

  return std::make_pair(row, col);
}

}  // namespace

Board::Board(const std::vector<PiecesOnBoard>& pieces) {
  board_.fill(0);

  for (auto& [piece, pos] : pieces) {
    for (std::string_view notation : pos) {
      auto [row, col] = ChessNotationToCoord(notation);
      PutPieceAt(row, col, Piece(piece));
    }
  }
}

Piece Board::PieceAt(int row, int col) const {
  const int index = 4 * (8 * row + col) / 64;
  const int offset_in_board_elem = 4 * (8 * row + col) % 64;

  // Extract 4 bit information about the piece.
  uint8_t info = (board_[index] & (0b1111ll << offset_in_board_elem)) >>
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
