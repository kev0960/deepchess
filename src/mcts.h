#ifndef MCTS_H
#define MCTS_H

#include <memory>

#include "dirichlet.h"
#include "evaluator.h"
#include "mcts_node.h"

namespace chess {

class MCTS {
 public:
  // Create MCTS with starting game state.
  MCTS(const GameState* state, Evaluator* evaluator,
       DirichletDistribution* dirichlet_dist);

  void RunMCTS(int num_iteration);

  // Get the policy vector. Policy vector is the flattened 1d vector of 73 * 8
  // * 8 (= 1 * 4672).
  torch::Tensor GetPolicyVector() const;

  // Return the move that corresponds to most visited node.
  Move MoveToMake() const;

  void DumpDebugInfo() const;

 private:
  // Select the node to expand.
  MCTSNode* Select();

  // Expand the leaf node.
  void Expand(MCTSNode* node);

  // Evaluate the node and return value estimate of the node.
  float Evaluate(const MCTSNode* node);

  // Backup starting from the leaf node with the value.
  void Backup(MCTSNode* leaf_node);

  void DumpDebugInfo(MCTSNode* node, int depth) const;

  std::vector<std::unique_ptr<MCTSNode>> nodes_;

  MCTSNode* root_;
  Evaluator* evaluator_;
  DirichletDistribution* dirichlet_dist_;

  int current_iter_ = 0;
};

}  // namespace chess

#endif
