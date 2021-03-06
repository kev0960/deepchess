#include "nn_util.h"

namespace chess {

void SetPieceOnTensor(const Board& board, int n_th, int row, int col,
                      PieceSide me, torch::Tensor* tensor) {
  const Piece piece = board.PieceAt(row, col);
  if (piece.Type() == PieceType::EMPTY) {
    return;
  }

  int index = 0;
  if (me != piece.Side()) {
    index += 6;
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

  int write_pos = 14 * n_th + index;
  tensor->index_put_({write_pos, row, col}, 1);
}

// Needs 8 previous board states. (Newest is the last element).
torch::Tensor BoardToTensor(const std::vector<Board>& boards,
                            PieceSide my_side) {
  torch::Tensor tensor = torch::zeros({117, 8, 8});

  // Scan entire board and construct the board.
  int count = 0;
  for (auto itr = boards.rbegin(); itr != boards.rend(); itr++) {
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        SetPieceOnTensor(*itr, count, row, col, my_side, &tensor);
      }
    }

    count++;
    if (count >= 8) {
      break;
    }
  }

  return tensor;
}

}  // namespace chess
