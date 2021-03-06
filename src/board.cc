#include "board.h"

#include <fmt/core.h>

#include "bit_util.h"
#include "piece_moves/bishop.h"
#include "piece_moves/king.h"
#include "piece_moves/knight.h"
#include "piece_moves/pawn.h"
#include "piece_moves/queen.h"
#include "piece_moves/rook.h"

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

Board::Board() { board_.fill(0); }

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

Piece Board::PieceAt(std::pair<int, int> coord) const {
  return PieceAt(coord.first, coord.second);
}

std::vector<Move> Board::GetAvailableMoves() const {
  std::vector<Move> moves;

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      std::vector<Move> piece_moves = GetMoveOfPieceAt(i, j);
      moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
    }
  }

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

void Board::PutPieceAt(std::pair<int, int> coord, Piece piece) {
  PutPieceAt(coord.first, coord.second, piece);
}

bool Board::IsEmptyAt(int row, int col) const {
  return PieceAt(row, col).Type() == EMPTY;
}

std::vector<Move> Board::GetMoveOfPieceAt(int row, int col) const {
  Piece piece = PieceAt(row, col);
  switch (piece.Type()) {
    case PAWN:
      return PawnMove::GetMoves(*this, piece, row, col);
    case ROOK:
      return RookMove::GetMoves(*this, piece, row, col);
    case KNIGHT:
      return KnightMove::GetMoves(*this, piece, row, col);
    case BISHOP:
      return BishopMove::GetMoves(*this, piece, row, col);
    case QUEEN:
      return QueenMove::GetMoves(*this, piece, row, col);
    case KING:
      return KingMove::GetMoves(*this, piece, row, col);
    default:
      return {};
  }
}

std::vector<Move> Board::GetMoveOfPieceAt(std::string_view coord) const {
  auto [row, col] = ChessNotationToCoord(coord);
  return GetMoveOfPieceAt(row, col);
}

Board Board::DoMove(Move m) const {
  Board next(*this);

  // Mark as empty.
  next.PutPieceAt(m.ToCoord(), PieceAt(m.FromCoord()));
  next.PutPieceAt(m.FromCoord(), Piece(" "));

  return next;
}

uint64_t Board::GetBinaryAvailableMoveOf(PieceSide side) const {
  uint64_t binary_board = 0;
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      Piece piece = PieceAt(row, col);

      if (piece.Side() != side) {
        continue;
      }

      auto moves = GetMoveOfPieceAt(row, col);
      for (auto m : moves) {
        binary_board = OnBitAt(binary_board, m.To());
        fmt::print("{} {} \n", piece.Print(), m.ToStr());
      }
    }
  }

  return binary_board;
}

uint64_t Board::GetBinaryPositionOfAll() const {
  uint64_t binary_board = 0;
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      Piece piece = PieceAt(row, col);
      if (piece.Type() == PieceType::EMPTY) {
        continue;
      }
      binary_board = OnBitAt(binary_board, row * 8 + col);
    }
  }

  return binary_board;
}

}  // namespace chess
