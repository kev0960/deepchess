#include "test_utils.h"

#include <fmt/core.h>

namespace chess {

GameStateBuilder& GameStateBuilder::DoMove(Move move) {
  states_.push_back(std::make_unique<GameState>(states_.back().get(), move));

  return *this;
}

Board BoardFromNotation(std::string_view notation) {
  Board board;

  notation.remove_prefix(
      std::min(notation.find_first_not_of('\n'), notation.size()));

  std::vector<std::string> rows = absl::StrSplit(notation, "\n");

  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < rows[row].size(); col++) {
      PieceType piece_type = PieceType::EMPTY;
      switch (std::tolower(rows[row][col])) {
        case 'p':
          piece_type = PieceType::PAWN;
          break;
        case 'n':
          piece_type = PieceType::KNIGHT;
          break;
        case 'b':
          piece_type = PieceType::BISHOP;
          break;
        case 'r':
          piece_type = PieceType::ROOK;
          break;
        case 'q':
          piece_type = PieceType::QUEEN;
          break;
        case 'k':
          piece_type = PieceType::KING;
          break;
      }

      PieceSide side = PieceSide::WHITE;
      if (std::islower(rows[row][col])) {
        side = PieceSide::BLACK;
      }

      board.PutPieceAt(row, col, Piece(piece_type, side));
    }
  }
  return board;
}
}  // namespace chess

