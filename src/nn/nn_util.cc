#include "nn_util.h"

namespace chess {
namespace {

using ::at::indexing::None;
using ::at::indexing::Slice;

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
        default:
          break;
      }

      tensor->index_put_({index + n_th * 14, row, col}, 1);
    }
  }
}

void SetRepititionsOnTensor(const GameState& game_state, int n_th,
                            torch::Tensor* tensor) {
  if (game_state.RepititionCount() >= 2) {
    // tensor[14::] = 1
    tensor->index_put_({Slice(14 * n_th + 12, None, None)}, 1);
  }

  if (game_state.RepititionCount() >= 3) {
    // tensor[14::] = 1
    tensor->index_put_({Slice(14 * n_th + 13, None, None)}, 1);
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
    tensor->index_put_({Slice(14 * 8 + 2, None, None)}, 1);
  }
  if (castling.second) {
    tensor->index_put_({Slice(14 * 8 + 3, None, None)}, 1);
  }

  if (side == PieceSide::BLACK) {
    castling = game_state.CanWhiteCastle();
  } else {
    castling = game_state.CanBlackCastle();
  }

  // Set P2 Castling.
  if (castling.first) {
    tensor->index_put_({Slice(14 * 8 + 4, None, None)}, 1);
  }
  if (castling.second) {
    tensor->index_put_({Slice(14 * 8 + 5, None, None)}, 1);
  }
}

void SetAuxiliaryData(const GameState& game_state, PieceSide side,
                      torch::Tensor* tensor) {
  if (side == PieceSide::BLACK) {
    tensor->index_put_({Slice(14 * 8, None, None)}, 1);
  }

  tensor->index_put_({Slice(14 * 8 + 1, None, None)},
                     game_state.TotalMoveCount());

  SetCastling(game_state, side, tensor);

  tensor->index_put_({Slice(14 * 8 + 6, None, None)},
                     game_state.NoProgressCount());
}

}  // namespace

// Needs 8 previous board states. (Newest is the last element).
torch::Tensor GameStateToTensor(const GameState& current_state,
                                PieceSide my_side) {
  torch::Tensor tensor = torch::zeros({117, 8, 8});

  // Scan entire board and construct the board.
  int n_th = 0;
  const GameState* current = &current_state;
  while (current) {
    SetPieceOnTensor(current->GetBoard(), my_side, n_th, &tensor);
    SetRepititionsOnTensor(*current, n_th, &tensor);

    n_th++;
    if (n_th >= 8) {
      break;
    }

    current = current->PrevState();
  }

  SetAuxiliaryData(current_state, my_side, &tensor);
  return tensor;
}

}  // namespace chess
