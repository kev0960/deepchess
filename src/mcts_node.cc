#include "mcts_node.h"

#include <fmt/core.h>

#include <cmath>

namespace chess {

MCTSNode::MCTSNode(std::unique_ptr<GameState> game_state, MCTSNode* parent,
                   float prior)
    : state_(std::move(game_state)),
      parent_(parent),
      w_s_a_(0),
      v_(0),
      prior_(prior),
      n_s_a_(0) {}

void MCTSNode::UpdateQ(float value) {
  w_s_a_ += value;
  n_s_a_ += 1;
}

void MCTSNode::SetValueOfThisState(float value) {
  v_ = value;
  computed_ = true;
}

void MCTSNode::AddChildNode(MCTSNode* node, const Move& move) {
  next_state_actions_.push_back(std::make_pair(node, move));
}

std::vector<std::pair<MCTSNode*, Move>>& MCTSNode::Children() {
  return next_state_actions_;
}

float MCTSNode::PUCT(int total_visit) const {
  return prior_ * std::sqrt(total_visit) / (1 + Visit());
}

const GameState& MCTSNode::State() const { return *state_; }

MCTSNode* MCTSNode::Parent() const { return parent_; }

int MCTSNode::Visit() const { return n_s_a_ + virtual_visit_; }

float MCTSNode::Q() const { return w_s_a_ / Visit() + virtual_loss_; }

float MCTSNode::V() const { return v_; }

void MCTSNode::AddVirtualLoss(float loss) {
  virtual_loss_ += loss;
  virtual_visit_++;
}

void MCTSNode::ClearVirtualLoss() {
  virtual_loss_ = 0;
  virtual_visit_ = 0;
}

float MCTSNode::Prior() const { return prior_; }

void MCTSNode::DumpDebugInfo() const {
  if (n_s_a_ != 0) {
    fmt::print(
        "W(s,a)=[{}] N(s,a)=[{}] Q(s,a)=[{}] Value=[{}] Prior=[{}]\n",
        w_s_a_, n_s_a_, w_s_a_ / n_s_a_, v_, prior_);
  } else {
    fmt::print("W(s,a)=[{}] N(s,a)=[{}] Value=[{}] Prior=[{}] \n",
               w_s_a_, n_s_a_, v_, prior_);
  }
}

}  // namespace chess
