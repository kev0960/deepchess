#include "mcts.h"

#include "nn/nn_util.h"

namespace chess {

MCTS::MCTS(const GameState* state, Evaluator* evaluator)
    : evaluator_(evaluator) {
  nodes_.push_back(std::make_unique<MCTSNode>(
      std::make_unique<GameState>(*state), /*parent=*/nullptr, /*prior=*/1));

  root_ = nodes_.back().get();
}

// Run selection - eval - expand - backup once.
void MCTS::RunMCTS() {
  MCTSNode* leaf = Select();

  Expand(leaf);

  // Evaluate the current position from the perspective of the current player of
  // leaf. If it is good, then it means it is bad for the previous player. So
  // when we backpropagate, we alternate the sign of q.
  float q = Evaluate(leaf);
  leaf->SetValueOfThisState(q);

  Backup(leaf);
}

// Select the leaf node to expand.
MCTSNode* MCTS::Select() {
  MCTSNode* current = root_;

  // Find the leaf node.
  while (true) {
    if (current->Children().empty()) {
      break;
    }

    // If not empty, then find the one with the largest Q + U.
    MCTSNode* max_elem = nullptr;
    float max_score = -10000000;

    for (auto& [node, action] : current->Children()) {
      if (node->Visit() == 0) {
        max_elem = node;
        break;
      }

      // Compute Q(s,a) + U(s,a). Note that the state "node" represents is s',
      // not s.
      float q_plus_u = node->PUCT(current->Visit()) + node->Q();
      if (max_elem == nullptr || q_plus_u > max_score) {
        max_elem = node;
        max_score = q_plus_u;
      }
    }

    current = max_elem;
  }

  return current;
}

void MCTS::Expand(MCTSNode* node) {
  // Expand the node by adding the child (node, actions).
  const GameState& state = node->State();

  // If current state is draw, then it is over.
  if (state.IsDraw()) { return ;}

  std::vector<Move> possible_moves = state.GetLegalMoves();

  // TODO Also need to consider king castling.
  for (const Move& move : possible_moves) {
    nodes_.push_back(std::make_unique<MCTSNode>(
        std::make_unique<GameState>(&state, move), node, /*prior=*/1));
    node->AddChildNode(nodes_.back().get(), move);
  }
}

float MCTS::Evaluate(const MCTSNode* node) {
  return evaluator_->Evalulate(node->State());
}

void MCTS::Backup(MCTSNode* leaf_node) {
  // Negate the value estimate as this is measured from the perspective of
  // curret node's player. However, Q(s,a) is computed from the perspective of
  // previous player. So we simply negate the value.
  float q = -leaf_node->V();
  MCTSNode* current = leaf_node;

  while (current) {
    current->UpdateQ(q);
    current = current->Parent();

    // Since the player alternates by the state, we have to negate the sign
    // every time.
    q = q * -1;
  }
}

torch::Tensor MCTS::GetPolicyVector() const {
  std::vector<std::pair<Move, float>> move_and_prob;
  move_and_prob.reserve(root_->Children().size());

  float total_visit = 0;
  for (const auto& [child_node, move] : root_->Children()) {
    move_and_prob.push_back(std::make_pair(move, child_node->Visit()));
    total_visit += child_node->Visit();
  }

  // Now we normalize the visit count.
  for (auto& [move, prob] : move_and_prob) {
    prob = prob / total_visit;
  }

  torch::Tensor policy = MoveToTensor(move_and_prob);

  // Return it as a 1d vector.
  return policy.flatten(0);
}

Move MCTS::MoveToMake() const {
  std::optional<Move> best_move;
  int max_visit = 0;

  for (const auto& [child_node, move] : root_->Children()) {
    if (child_node->Visit() > max_visit) {
      max_visit = child_node->Visit();
      best_move = move;
    }
  }

  assert(best_move.has_value());
  return best_move.value();
}

}  // namespace chess
