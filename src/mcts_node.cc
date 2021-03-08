#include "mcts_node.h"

#include <cmath>

namespace chess {

MCTSNode::MCTSNode(const GameState& game_state, MCTSNode* parent, float prior)
    : state_(game_state),
      parent_(parent),
      value_(0),
      prior_(prior),
      visit_(0) {}

void MCTSNode::UpdateWithValue(float value) {
  value_ += value;
  visit_ += 1;
}

void MCTSNode::AddChildNode(MCTSNode* node, const Move& move) {
  next_state_actions_.push_back(std::make_pair(node, move));
}

std::vector<std::pair<MCTSNode*, Move>>& MCTSNode::Children() {
  return next_state_actions_;
}

float MCTSNode::PUCT(int total_visit) const {
  return prior_ * std::sqrt(total_visit) / (1 + visit_);
}

const GameState& MCTSNode::State() const { return state_; }

MCTSNode* MCTSNode::Parent() const { return parent_; }

int MCTSNode::Visit() const { return visit_; }

float MCTSNode::Q() const { return value_ / visit_; }

}  // namespace chess
