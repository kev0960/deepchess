#ifndef NN_NN_UTIL_H
#define NN_NN_UTIL_H

#include <torch/torch.h>

#include <vector>

#include "board.h"

namespace chess {

// Convert board to tensor.
torch::Tensor BoardToTensor(const std::vector<Board>& board, PieceSide my_side);

}  // namespace chess

#endif
