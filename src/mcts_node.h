#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <memory>
#include <utility>
#include <vector>

#include "game_state.h"
#include "move.h"

namespace chess {

// Node of MCTS.
//
// Note that each MCTS node *owns* the GameState that it represents.
class MCTSNode {
 public:
  MCTSNode(std::unique_ptr<GameState> state, MCTSNode* parent, float prior);

  // Update the Q(s,a) where s is the previous state.
  void UpdateQ(float value);

  // Set the value of this state (from the estimiation of NN)
  void SetValueOfThisState(float value);

  // Add a child node.
  void AddChildNode(MCTSNode* node, const Move& move);

  std::vector<std::pair<MCTSNode*, Move>>& Children();

  // Compute PUCT score of this node.
  float PUCT(int total_visit) const;

  // Get the state represented by this node.
  const GameState& State() const;

  MCTSNode* Parent() const;

  // Total number of visit of this node.
  int Visit() const;

  // Returns Q(s,a), which is W(s,a) / N(s,a). Note that s is a previous state.
  float Q() const;

  // Value estimate of the current state.
  float V() const;

 private:
  // State.
  std::unique_ptr<GameState> state_;

  MCTSNode* parent_;

  // W(s,a) where s is the previous state.
  float w_s_a_;

  // Value of this node estimated by the neural net.
  float v_;

  // Prior probability.
  float prior_;

  // N(s, a) where s is the previous node.
  int n_s_a_;

  // Child nodes and actions.
  std::vector<std::pair<MCTSNode*, Move>> next_state_actions_;
};

}  // namespace chess

#endif
