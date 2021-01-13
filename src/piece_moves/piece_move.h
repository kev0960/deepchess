#ifndef PIECE_MOVES_PIECE_MOVE_H
#define PIECE_MOVES_PIECE_MOVE_H

#include "board.h"
#include "piece.h"

namespace chess {

void FetchMoveFromDelta(const Board& board, const Piece& piece,
                        std::vector<Move>& moves, int row, int col,
                        int delta_row, int delta_col, int num_step = 7);

}  // namespace chess

#endif
