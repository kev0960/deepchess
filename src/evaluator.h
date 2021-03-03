#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "board.h"

namespace chess {

class Evaluator {
 public:
  virtual float Evalulate(const Board& board);
};

}  // namespace chess

#endif
