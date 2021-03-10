#ifndef MCTS_H
#define MCTS_H

#include <memory>

#include "evaluator.h"
#include "mcts_node.h"

namespace chess {

class MCTS {
 public:
  // Create MCTS with starting game state.
  MCTS(const GameState* state, Evaluator* evaluator);

  void RunMCTS();

  // Get the policy vector. Policy vector is the flattened 1d vector of 73 * 8
  // * 8 (= 1 * 4672).
  torch::Tensor GetPolicyVector();

 private:
  // Select the node to expand.
  MCTSNode* Select();

  // Expand the leaf node.
  void Expand(MCTSNode* node);

  // Evaluate the node and return Q value.
  float Evaluate(const MCTSNode* node);

  // Backup starting from the leaf node with the Q value.
  void Backup(MCTSNode* leaf_node, float q);

  std::vector<std::unique_ptr<MCTSNode>> nodes_;

  MCTSNode* root_;
  Evaluator* evaluator_;
};

}  // namespace chess

#endif
