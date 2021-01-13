#ifndef PIECE_MOVES_ROOK_H
#define PIECE_MOVES_ROOK_H

#include "board.h"
#include "move.h"

namespace chess {

class RookMove {
 public:
  static std::vector<Move> GetMoves(const Board& board, const Piece& piece,
                                    int row, int col);
};

}  // namespace chess

#endif
