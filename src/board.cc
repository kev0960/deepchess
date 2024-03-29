#include "board.h"

#include <fmt/core.h>

#include "bit_util.h"
#include "move.h"
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

bool IsKingOkay(const Board& board, PieceSide opponent, Move move,
                int king_pos) {
  Board b = board.DoMove(move);

  if (move.From() == king_pos) {
    king_pos = move.To();
  }

  auto moves = b.GetAvailableMoves(opponent);
  for (auto m : moves) {
    if (m.To() == king_pos) {
      return false;
    }
  }

  return true;
}

Piece GetPromotedPiece(Promotion promo, PieceSide side) {
  switch (promo) {
    case PROMOTE_QUEEN:
      return Piece(QUEEN, side);
    case PROMOTE_KNIGHT:
      return Piece(KNIGHT, side);
    case PROMOTE_BISHOP:
      return Piece(BISHOP, side);
    case PROMOTE_ROOK:
      return Piece(ROOK, side);
    default:
      // TODO Mark this as an error.
      return Piece(PAWN, side);
  }
}

Board DoCastling(const Board& board, Move m) {
  Board next(board);

  Piece king = board.PieceAt(m.FromCoord());
  Piece rook(PieceType::ROOK, king.Side());
  Piece empty(" ");

  if (m.FromCoord().second < m.ToCoord().second) {
    // King side castling.
    next.PutPieceAt(m.ToCoord(), king);
    next.PutPieceAt(m.FromCoord(), empty);

    next.PutPieceAt(m.FromCoord().first, 5, rook);
    next.PutPieceAt(m.FromCoord().first, 7, empty);
  } else {
    // Queen side castling.
    next.PutPieceAt(m.ToCoord(), king);
    next.PutPieceAt(m.FromCoord(), empty);

    next.PutPieceAt(m.FromCoord().first, 3, rook);
    next.PutPieceAt(m.FromCoord().first, 0, empty);
  }

  return next;
}

int FindKing(const Board& board, PieceSide color) {
  int king_pos = 0;

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if (board.PieceAt(i, j).Side() == color &&
          board.PieceAt(i, j).Type() == KING) {
        // TODO Find more generic way to match it to Move's To and From.
        king_pos = i * 8 + j;
        break;
      }
    }
  }

  return king_pos;
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

std::vector<Move> Board::GetAvailableMoves(PieceSide who) const {
  std::vector<Move> moves;

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if (PieceAt(i, j).Side() == who && PieceAt(i, j).Type() != EMPTY) {
        std::vector<Move> piece_moves = GetMoveOfPieceAt(i, j);
        moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
      }
    }
  }

  return moves;
}

std::vector<Move> Board::GetAvailableLegalMoves(PieceSide me) const {
  std::vector<Move> moves;

  int king_pos = FindKing(*this, me);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if (PieceAt(i, j).Side() != me || PieceAt(i, j).Type() == EMPTY) {
        continue;
      }

      std::vector<Move> piece_moves = GetMoveOfPieceAt(i, j);
      for (auto m : piece_moves) {
        if (IsKingOkay(*this, me == BLACK ? WHITE : BLACK, m, king_pos)) {
          moves.push_back(m);
        }
      }
    }
  }

  return moves;
}

std::string Board::PrintBoard(char empty) const {
  std::string board;
  board.reserve(64 + 8);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      board.push_back(PieceAt(i, j).Print(empty));
    }
    board.push_back('\n');
  }

  return board;
}

void Board::PrettyPrintBoard() const {
  fmt::print("+---+---+---+---+---+---+---+---+\n");
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      fmt::print("| {} ", PieceAt(row, col).Print());
    }
    fmt::print("| {}\n", 8 - row);
    fmt::print("+---+---+---+---+---+---+---+---+\n");
  }
  fmt::print("  a   b   c   d   e   f   g   h\n");
}

std::string Board::PrintNumericBoard() const {
  std::string b;
  for (size_t i = 0; i < board_.size(); i++) {
    if (i != 0) {
      b.push_back(',');
    }
    b += std::to_string(board_[i]);
  }

  return b;
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

  Piece piece = PieceAt(m.FromCoord());
  if (m.GetPromotion() != NO_PROMOTE) {
    piece = GetPromotedPiece(m.GetPromotion(), piece.Side());
  }

  // Check if it is a castling.
  if (piece.Type() == PieceType::KING) {
    // If the king has moved two blocks, then it is a castling!
    if (std::abs(m.ToCoord().second - m.FromCoord().second) == 2) {
      return DoCastling(*this, m);
    }
  }

  // Check for En Passant if it was a diagonal move.
  if (piece.Type() == PieceType::PAWN &&
      std::abs(m.ToCoord().second - m.FromCoord().second) == 1) {
    if (PieceAt(m.ToCoord()).Type() == PieceType::EMPTY) {
      // Capture the passed pawn.
      next.PutPieceAt(m.FromCoord().first, m.ToCoord().second, Piece(" "));
    }
  }

  // Mark as empty.
  next.PutPieceAt(m.ToCoord(), piece);
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

bool Board::operator==(const Board& board) const {
  return board_ == board.board_;
}

bool Board::operator!=(const Board& board) const {
  return board_ != board.board_;
}

bool Board::IsCheck(PieceSide color) const {
  int king_pos = FindKing(*this, color);

  auto moves = GetAvailableMoves(color == WHITE ? BLACK : WHITE);
  for (auto m : moves) {
    if (m.To() == king_pos) {
      return true;
    }
  }

  return false;
}

bool Board::DrawByInsufficientMaterial() const {
  int knight_count = 0, bishop_count = 0;

  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      PieceType piece = PieceAt(row, col).Type();
      if (piece == EMPTY || piece == KING) {
        continue;
      }

      if (piece == KNIGHT) {
        knight_count++;
      } else if (piece == BISHOP) {
        bishop_count++;
      } else {
        return false;
      }
    }
  }

  // King and bishop versus King or
  // King and knight versus King or
  // King versus King
  if (knight_count + bishop_count <= 1) {
    return true;
  }

  return false;
}

}  // namespace chess
