#ifndef PIECE_MOVES_KNIGHT_H
#define PIECE_MOVES_KNIGHT_H

#include "board.h"
#include "move.h"

namespace chess {

class KnightMove {
 public:
  static std::vector<Move> GetMoves(const Board& board, const Piece& piece,
                                    int row, int col);
};

}  // namespace chess

#endif

