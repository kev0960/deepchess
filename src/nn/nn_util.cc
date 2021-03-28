#include "nn_util.h"

namespace chess {
namespace {

using ::at::indexing::None;
using ::at::indexing::Slice;

constexpr int kNumFeaturesPerHistory = 14;
constexpr int kNumMaxHistory = 8;
constexpr int kTotalNumFeatures = kNumFeaturesPerHistory * kNumMaxHistory + 7;

constexpr int kQueenMoveN = 0;
constexpr int kQueenMoveNE = 1 * 7;
constexpr int kQueenMoveE = 2 * 7;
constexpr int kQueenMoveSE = 3 * 7;
constexpr int kQueenMoveS = 4 * 7;
constexpr int kQueenMoveSW = 5 * 7;
constexpr int kQueenMoveW = 6 * 7;
constexpr int kQueenMoveNW = 7 * 7;

// Create 12 * 8 * 8 tensor that encodes P1 piece and P2 piece.
void SetPieceOnTensor(const Board& board, PieceSide me, int n_th,
                      torch::Tensor* tensor) {
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      Piece piece = board.PieceAt(row, col);

      int index = 0;
      if (me != piece.Side()) {
        index = 6;
      }

      switch (piece.Type()) {
        case PieceType::PAWN:
          break;
        case PieceType::KNIGHT:
          index += 1;
          break;
        case PieceType::BISHOP:
          index += 2;
          break;
        case PieceType::ROOK:
          index += 3;
          break;
        case PieceType::QUEEN:
          index += 4;
          break;
        case PieceType::KING:
          index += 5;
          break;
        case PieceType::EMPTY:
          continue;
      }

      tensor->index_put_({index + n_th * kNumFeaturesPerHistory, row, col}, 1);
    }
  }
}

void SetRepititionsOnTensor(const GameState& game_state, int n_th,
                            torch::Tensor* tensor) {
  if (game_state.RepititionCount() >= 2) {
    tensor->index_put_({kNumFeaturesPerHistory * n_th + 12}, 1);
  }

  if (game_state.RepititionCount() >= 3) {
    tensor->index_put_({kNumFeaturesPerHistory * n_th + 13}, 1);
  }
}

void SetCastling(const GameState& game_state, PieceSide side,
                 torch::Tensor* tensor) {
  std::pair<bool, bool> castling;
  if (side == PieceSide::BLACK) {
    castling = game_state.CanBlackCastle();
  } else {
    castling = game_state.CanWhiteCastle();
  }

  // Set P1 Castling.
  if (castling.first) {
    tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 2}, 1);
  }
  if (castling.second) {
    tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 3}, 1);
  }

  if (side == PieceSide::BLACK) {
    castling = game_state.CanWhiteCastle();
  } else {
    castling = game_state.CanBlackCastle();
  }

  // Set P2 Castling.
  if (castling.first) {
    tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 4}, 1);
  }
  if (castling.second) {
    tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 5}, 1);
  }
}

void SetAuxiliaryData(const GameState& game_state, PieceSide side,
                      torch::Tensor* tensor) {
  if (side == PieceSide::BLACK) {
    tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory}, 1);
  }

  tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 1},
                     game_state.TotalMoveCount());
  SetCastling(game_state, side, tensor);
  tensor->index_put_({kNumFeaturesPerHistory * kNumMaxHistory + 6},
                     game_state.NoProgressCount());
}

int ComputeTensorSize(c10::IntArrayRef ref) {
  int total = 1;
  for (auto i : ref) {
    total *= i;
  }

  return total;
}

int GetMoveIndex(Move m) {
  auto [from_row, from_col] = m.FromCoord();
  auto [to_row, to_col] = m.ToCoord();

  // Queen promotion is recored as a regular pawn move th last row.
  if (m.GetPromotion() != NO_PROMOTE && m.GetPromotion() != PROMOTE_QUEEN) {
    int diff = 1 + to_col - from_col;
    switch (m.GetPromotion()) {
      case PROMOTE_KNIGHT:
        return 64 + diff;
      case PROMOTE_BISHOP:
        return 64 + 3 + diff;
      case PROMOTE_ROOK:
        return 64 + 6 + diff;
      default:
        // TODO Put assert false.
        return -1;
    }
  }

  // Diagonal Queen move.
  if (std::abs(from_row - to_row) == std::abs(from_col - to_col)) {
    int dist = std::abs(from_row - to_row) - 1;
    if (from_col < to_col) {
      if (from_row < to_row) {
        return kQueenMoveSE + dist;
      } else {
        return kQueenMoveNE + dist;
      }
    } else {
      if (from_row < to_row) {
        return kQueenMoveSW + dist;
      } else {
        return kQueenMoveNW + dist;
      }
    }
  } else if (from_row == to_row) {
    // Horizontal Queen move.
    if (from_col < to_col) {
      return kQueenMoveE + (to_col - from_col - 1);
    } else {
      return kQueenMoveW + (from_col - to_col - 1);
    }
  } else if (from_col == to_col) {
    // Vertical Queen move.
    if (from_row < to_row) {
      return kQueenMoveS + (to_row - from_row - 1);
    } else {
      return kQueenMoveN + (from_row - to_row - 1);
    }
  }

  // Otherwise it is a knight move.
  static int knight_arr[5][2] = {{0, 1}, {2, 3}, {0, 0}, {4, 5}, {6, 7}};

  return 56 + knight_arr[to_row - from_row + 2][to_col > from_col ? 0 : 1];
}
void SetPolicyTensor(Move m, float prob, torch::Tensor* tensor) {
  auto [from_row, from_col] = m.FromCoord();
  tensor->index_put_({GetMoveIndex(m), from_row, from_col}, prob);
}

}  // namespace

// Needs 8 previous board states. (Newest is the last element).
torch::Tensor GameStateToTensor(const GameState& current_state) {
  torch::Tensor tensor = torch::zeros({kTotalNumFeatures, 8, 8});

  // Scan entire board and construct the board.
  int n_th = 0;
  const GameState* current = &current_state;
  while (current) {
    SetPieceOnTensor(current->GetBoard(), current_state.WhoIsMoving(), n_th,
                     &tensor);
    SetRepititionsOnTensor(*current, n_th, &tensor);

    n_th++;
    if (n_th >= kNumMaxHistory) {
      break;
    }

    current = current->PrevState();
  }

  SetAuxiliaryData(current_state, current_state.WhoIsMoving(), &tensor);

  return tensor;
}

int GetModelNumParams(ChessNN m) {
  int total_params = 0;
  for (auto& item : m->named_parameters()) {
    total_params += ComputeTensorSize(item.value().sizes());
  }

  return total_params;
}

torch::Tensor MoveToTensor(std::vector<std::pair<Move, float>> move_and_prob) {
  torch::Tensor policy = torch::zeros({73, 8, 8});

  for (auto& [m, p] : move_and_prob) {
    SetPolicyTensor(m, p, &policy);
  }

  return policy;
}

torch::Tensor NormalizePolicy(const GameState& game_state,
                              torch::Tensor policy) {
  std::vector<std::pair<Move, float>> move_and_prob_for_mask;
  for (const auto& move : game_state.GetLegalMoves()) {
    move_and_prob_for_mask.push_back(std::make_pair(move, 1));
  }

  torch::Tensor mask = MoveToTensor(move_and_prob_for_mask);
  mask = mask.to(policy.device());
  mask = mask.unsqueeze(0).flatten(1);

  policy = policy * mask;

  // We are adding small probability to every possible moves to avoid division
  // by zero (this might be a case where the valid moves are not populated in
  // the policy vector)
  mask = mask * 0.00001;

  policy = policy + mask;
  policy = policy / torch::sum(policy);

  return policy;
}
}  // namespace chess
