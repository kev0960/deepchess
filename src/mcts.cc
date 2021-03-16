#include "mcts.h"

#include "nn/nn_util.h"

namespace chess {
namespace {

float ComputePrior(float p_a, float dirchlet) {
  // Epsilon = 0.25.
  return 0.75 * p_a + 0.25 * dirchlet;
}

std::vector<std::vector<const GameState*>> CreateBatches(MCTSNode* node,
                                                         size_t batch_size,
                                                         size_t total_batchs) {
  std::vector<std::vector<const GameState*>> batches;
  size_t current_batch_num = 0;
  size_t child_index = 0;

  while (current_batch_num < total_batchs) {
    batches.emplace_back();

    std::vector<const GameState*>& batch = batches.back();
    while (batches.back().size() < batch_size) {
      if (child_index >= node->Children().size()) {
        return batches;
      }

      // Only add ones that are not computed to the batch.
      if (!node->Children()[child_index].first->Computed()) {
        batch.push_back(&node->Children()[child_index].first->State());
      }

      child_index++;
    }
  }

  return batches;
}

void PreComputeBatches(
    MCTSNode* current, Evaluator* evaluator,
    const std::vector<std::vector<const GameState*>>& batches) {
  size_t child_index = 0;
  size_t batch_index = 0;

  while (batch_index < batches.size()) {
    std::vector<float> values = evaluator->EvalulateBatch(batches[batch_index]);

    size_t i = 0;
    while (i < values.size()) {
      if (current->Children()[child_index].first->Computed()) {
        child_index++;
        continue;
      }

      current->Children()[child_index].first->SetValueOfThisState(values[i]);
      i++;
      child_index++;
    }

    batch_index++;
  }
}

}  // namespace

MCTS::MCTS(const GameState* state, Evaluator* evaluator,
           DirichletDistribution* dirichlet_dist)
    : evaluator_(evaluator), dirichlet_dist_(dirichlet_dist) {
  nodes_.push_back(std::make_unique<MCTSNode>(
      std::make_unique<GameState>(*state), /*parent=*/nullptr, /*prior=*/1));

  root_ = nodes_.back().get();
}

// Run selection - eval - expand - backup once.
void MCTS::RunMCTS(int num_iteration) {
  for (; current_iter_ < num_iteration; current_iter_++) {
    MCTSNode* leaf = Select();

    Expand(leaf);

    // Evaluate the current position from the perspective of the current player
    // of leaf. If it is good, then it means it is bad for the previous player.
    // So when we backpropagate, we alternate the sign of q.
    float q = Evaluate(leaf);
    leaf->SetValueOfThisState(q);

    Backup(leaf);
  }
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
  if (state.IsDraw()) {
    return;
  }

  std::vector<Move> possible_moves = state.GetLegalMoves();
  std::vector<float> dirichlet_dist =
      dirichlet_dist_->GetDistribution(possible_moves.size());

  for (size_t i = 0; i < possible_moves.size(); i++) {
    const Move& move = possible_moves[i];
    float noise = dirichlet_dist[i];

    nodes_.push_back(
        std::make_unique<MCTSNode>(std::make_unique<GameState>(&state, move),
                                   node, ComputePrior(node->Prior(), noise)));
    node->AddChildNode(nodes_.back().get(), move);
  }

  // For the root node, evey child will be visited anyway. So we just batch run
  // every nodes.
  if (root_ == node) {
    std::vector<std::vector<const GameState*>> batches =
        CreateBatches(node, 20, possible_moves.size());
    PreComputeBatches(node, evaluator_, batches);
  } else if (node->Parent()->Visit() >= 2) {
    // If the parent was visited more than 2 times before, then it is likely
    // that every child node of this parent will get visited too. Hence let's
    // just precompute all the values of child.
    std::vector<std::vector<const GameState*>> batches =
        CreateBatches(node->Parent(), 20, 20);
    PreComputeBatches(node->Parent(), evaluator_, batches);
  }
}

float MCTS::Evaluate(const MCTSNode* node) {
  if (node->Computed()) {
    return node->V();
  }

  // std::cout << "Evaluating " << std::endl;
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

  if (!best_move.has_value()) {
    root_->State().GetBoard().PrettyPrintBoard();
    fmt::print("Very Weird!");
    DumpDebugInfo();
    std::cout << std::endl;
    assert(best_move.has_value());
  }
  return best_move.value();
}

void MCTS::DumpDebugInfo() const { DumpDebugInfo(root_, 0); }

void MCTS::DumpDebugInfo(MCTSNode* node, int depth) const {
  if (node->Visit() == 0) {
    return;
  }

  if (node == root_) {
    fmt::print("{:02} Root ----\n", depth);
  } else {
    fmt::print("{:02} {}   ----\n", depth, node->State().LastMove().Str());
  }

  node->DumpDebugInfo();
  for (const auto& n : node->Children()) {
    DumpDebugInfo(n.first, depth + 1);
  }
}

}  // namespace chess
