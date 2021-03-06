#include "game_state.h"

namespace chess {

// Make sure that the king and the squares that the king passes is not under
// attack.
constexpr uint64_t kWhiteKingSideCastleAttackCheck = 0x70LL << 56;
constexpr uint64_t kWhiteQueenSideCastleAttackCheck = 0x1CLL << 56;
constexpr uint64_t kBlackKingSideCastleAttackCheck = 0x70;
constexpr uint64_t kBlackQueenSideCastleAttackCheck = 0x1C;

// Make sure that there is no obstacle between the rook and the king.
constexpr uint64_t kWhiteKingSideCastleMoveCheck = 0x60LL << 56;
constexpr uint64_t kWhiteQueenSideCastleMoveCheck = 0xCLL << 56;
constexpr uint64_t kBlackKingSideCastleMoveCheck = 0x60;
constexpr uint64_t kBlackQueenSideCastleMoveCheck = 0xC;

GameState::GameState(const Board& board)
    : current_board_(board), prev_state_(nullptr) {}

GameState::GameState(const GameState* prev_state, Move move)
    : current_board_(prev_state->GetBoard().DoMove(move)),
      white_castle_(prev_state->white_castle_),
      black_castle_(prev_state->black_castle_),
      prev_state_(prev_state) {
  if (move.FromCoord() == std::make_pair(0, 0)) {
    black_castle_.queen_side_rook_moved = true;
  } else if (move.FromCoord() == std::make_pair(0, 7)) {
    black_castle_.king_side_rook_moved = true;
  } else if (move.FromCoord() == std::make_pair(0, 4)) {
    black_castle_.king_moved = true;
  }

  if (move.FromCoord() == std::make_pair(7, 0)) {
    white_castle_.queen_side_rook_moved = true;
  } else if (move.FromCoord() == std::make_pair(7, 7)) {
    white_castle_.king_side_rook_moved = true;
  } else if (move.FromCoord() == std::make_pair(7, 4)) {
    white_castle_.king_moved = true;
  }
}

std::pair<bool, bool> GameState::CanWhiteCastle() const {
  if (white_castle_.king_moved) {
    return std::make_pair(false, false);
  }

  bool can_castle_king_side = true, can_castle_queen_side = true;

  if (white_castle_.king_side_rook_moved) {
    can_castle_king_side = false;
  } else if (white_castle_.queen_side_rook_moved) {
    can_castle_queen_side = false;
  }

  if (!can_castle_king_side && !can_castle_queen_side) {
    return std::make_pair(false, false);
  }

  // Check whether the king does not go through a square that is attacked. This
  // check includes whether the current king is being checked or not.
  uint64_t black_can_attack =
      current_board_.GetBinaryAvailableMoveOf(PieceSide::BLACK);
  if (can_castle_king_side) {
    can_castle_king_side = kWhiteKingSideCastleAttackCheck & black_can_attack;
  }

  if (can_castle_queen_side) {
    can_castle_queen_side = kWhiteQueenSideCastleAttackCheck & black_can_attack;
  }

  if (!can_castle_king_side && !can_castle_queen_side) {
    return std::make_pair(false, false);
  }

  // Now check whether there are any obstacle between king and the rook.
  uint64_t current_pieces = current_board_.GetBinaryPositionOfAll();
  if (can_castle_king_side) {
    can_castle_king_side = kWhiteKingSideCastleMoveCheck & current_pieces;
  }

  if (can_castle_queen_side) {
    can_castle_queen_side = kWhiteQueenSideCastleMoveCheck & current_pieces;
  }

  return std::make_pair(can_castle_king_side, can_castle_queen_side);
}

std::pair<bool, bool> GameState::CanBlackCastle() const {
  if (black_castle_.king_moved) {
    return std::make_pair(false, false);
  }

  bool can_castle_king_side = true, can_castle_queen_side = true;

  if (black_castle_.king_side_rook_moved) {
    can_castle_king_side = false;
  } else if (black_castle_.queen_side_rook_moved) {
    can_castle_queen_side = false;
  }

  if (!can_castle_king_side && !can_castle_queen_side) {
    return std::make_pair(false, false);
  }

  // Check whether the king does not go through a square that is attacked. This
  // check includes whether the current king is being checked or not.
  uint64_t white_can_attack =
      current_board_.GetBinaryAvailableMoveOf(PieceSide::WHITE);
  if (can_castle_king_side) {
    can_castle_king_side = kBlackKingSideCastleAttackCheck & white_can_attack;
  }

  if (can_castle_queen_side) {
    can_castle_queen_side = kBlackQueenSideCastleAttackCheck & white_can_attack;
  }

  if (!can_castle_king_side && !can_castle_queen_side) {
    return std::make_pair(false, false);
  }

  // Now check whether there are any obstacle between king and the rook.
  uint64_t current_pieces = current_board_.GetBinaryPositionOfAll();
  if (can_castle_king_side) {
    can_castle_king_side = kBlackKingSideCastleMoveCheck & current_pieces;
  }

  if (can_castle_queen_side) {
    can_castle_queen_side = kBlackQueenSideCastleMoveCheck & current_pieces;
  }

  return std::make_pair(can_castle_king_side, can_castle_queen_side);
}

GameState GameState::CreateInitGameState() {
  const static std::vector<PiecesOnBoard> pieces = {
      {"R", {"a1", "h1"}},
      {"N", {"b1", "g1"}},
      {"B", {"c1", "f1"}},
      {"Q", {"d1"}},
      {"K", {"e1"}},
      {"P", {"a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"}},
      {"r", {"a8", "h8"}},
      {"n", {"b8", "g8"}},
      {"b", {"c8", "f8"}},
      {"q", {"d8"}},
      {"k", {"e8"}},
      {"p", {"a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"}},
  };

  GameState state(Board{pieces});
  return state;
}

}  // namespace chess

