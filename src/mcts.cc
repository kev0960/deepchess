#include "mcts.h"

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

torch::Tensor MCTS::GetPolicyVector() {
  torch::Tensor policy = torch::zeros({73, 8, 8});

  // Return it as a 1d vector.
  return policy.flatten(0);
}

}  // namespace chess
