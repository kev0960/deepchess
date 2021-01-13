#ifndef PIECE_MOVES_KING_H
#define PIECE_MOVES_KING_H

#include "board.h"
#include "move.h"

namespace chess {

class KingMove {
 public:
  static std::vector<Move> GetMoves(const Board& board, const Piece& piece,
                                    int row, int col);
};

}  // namespace chess

#endif
