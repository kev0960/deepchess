#include "mcts.h"

namespace chess {

MCTS::MCTS(const Board& board, Evaluator* evaluator) : evaluator_(evaluator) {
  nodes_.push_back(
      std::make_unique<MCTSNode>(board, /*parent=*/nullptr, /*prior=*/1));

  root_ = nodes_.back().get();
}

// Run selection - eval - expand - backup once.
void MCTS::RunMCTS() {
  MCTSNode* leaf = Select();

  Expand(leaf);

  float q = Evaluate(leaf);
  Backup(leaf, q);
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
    float max_score = 0;

    for (auto& [node, action] : current->Children()) {
      if (node->Visit() == 0) {
        max_elem = node;
        break;
      }

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
  const Board& board = node->State();

  std::vector<Move> possible_moves = board.GetAvailableMoves();
  for (const Move& move : possible_moves) {
    nodes_.push_back(
        std::make_unique<MCTSNode>(board.DoMove(move), node, /*prior=*/1));
    node->AddChildNode(nodes_.back().get(), move);
  }
}

float MCTS::Evaluate(const MCTSNode* node) {
  return evaluator_->Evalulate(node->State());
}

void MCTS::Backup(MCTSNode* leaf_node, float q) {
  MCTSNode* current = leaf_node;
  while (current) {
    current->UpdateWithValue(q);
    current = current->Parent();
  }
}

}  // namespace chess
