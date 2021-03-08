#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <utility>
#include <vector>

#include "game_state.h"
#include "move.h"

namespace chess {

class MCTSNode {
 public:
  MCTSNode(const GameState& state, MCTSNode* parent, float prior);

  // If the node (or any child node) is visited, update the W value and increase
  // the visit count.
  void UpdateWithValue(float value);

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

  // Q score of this node (W / visit_)
  float Q() const;

 private:
  // State.
  const GameState state_;

  MCTSNode* parent_;

  // W(s)
  float value_;

  // Prior probability.
  float prior_;

  // N(s)
  int visit_;

  // Child nodes and actions.
  std::vector<std::pair<MCTSNode*, Move>> next_state_actions_;
};

}  // namespace chess

#endif
