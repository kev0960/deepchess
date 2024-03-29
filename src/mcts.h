#ifndef MCTS_H
#define MCTS_H

#include <memory>

#include "config.h"
#include "distribution.h"
#include "evaluator.h"
#include "mcts_node.h"

namespace chess {

class MCTS {
 public:
  // Create MCTS with starting game state.
  MCTS(const GameState* state, Evaluator* evaluator, Distribution* dist,
       Config* config, int worker_id);

  void RunMCTS();

  // Get the policy vector. Policy vector is the flattened 1d vector of 73 * 8
  // * 8 (= 1 * 4672).
  torch::Tensor GetPolicyVector() const;

  // Return the move that corresponds to most visited node.
  // If best_move is true, then it will select the move with the highest visit
  // count. Otherwise, it will sample the move based on the probability of each
  // move.
  Move MoveToMake(bool choose_best_move = false) const;

  void DumpDebugInfo() const;

  // Show the path from the root to the node.
  void ShowPath(MCTSNode* node) const;

 private:
  void DoSingleRun();
  void DoBatchRun();

  // Select the node to expand.
  MCTSNode* Select();

  // Expand the leaf node.
  void Expand(MCTSNode* node);

  // Evaluate the node and return value estimate of the node.
  float Evaluate(const MCTSNode* node);

  // Backup starting from the leaf node with the value.
  void Backup(MCTSNode* leaf_node);

  // Backup using the virtual loss.
  void BackupVirtual(MCTSNode* leaf_node, float virtual_loss);

  // Clear virtual loss set by leaf node.
  void ClearVirtual(MCTSNode* leaf_node);

  void DumpDebugInfo(MCTSNode* node, int depth) const;

  // NOTE: Since we shuffle the child nodes, the ordering of moves may be
  // different from the ordering returned from state.PossibleMoves().
  std::vector<std::unique_ptr<MCTSNode>> nodes_;

  MCTSNode* root_;
  Evaluator* evaluator_;
  Distribution* dist_;

  Config* config_;
  int current_iter_ = 0;
  int worker_id_;
};

}  // namespace chess

#endif
