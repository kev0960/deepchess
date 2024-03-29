#include "mcts.h"

#include <absl/strings/str_join.h>
#include <fmt/ranges.h>

#include "nn/nn_util.h"

namespace chess {
namespace {

float ComputePrior(float p_a, float dirchlet) {
  // Epsilon = 0.25.
  return 0.75 * p_a + 0.25 * dirchlet;
}

std::vector<std::vector<const GameState*>> CreateBatches(MCTSNode* node,
                                                         size_t batch_size,
                                                         size_t total_baches) {
  std::vector<std::vector<const GameState*>> batches;
  size_t current_batch_num = 0;
  size_t child_index = 0;

  while (current_batch_num < total_baches) {
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

MCTS::MCTS(const GameState* state, Evaluator* evaluator, Distribution* dist,
           Config* config, int worker_id)
    : evaluator_(evaluator),
      dist_(dist),
      config_(config),
      worker_id_(worker_id) {
  nodes_.push_back(std::make_unique<MCTSNode>(
      std::make_unique<GameState>(*state), /*parent=*/nullptr, /*prior=*/1));

  root_ = nodes_.back().get();
}

// Run selection - eval - expand - backup once.
void MCTS::RunMCTS() {
  if (config_->do_batch_mcts) {
    DoBatchRun();
  } else {
    DoSingleRun();
  }
}

void MCTS::DoBatchRun() {
  const int batch_leaf_size = config_->mcts_batch_leaf_node_size;
  while (current_iter_ < config_->num_mcts_iteration) {
    std::vector<MCTSNode*> batch_leaf_nodes;
    batch_leaf_nodes.reserve(batch_leaf_size);

    int current_batch_count = 0;
    while (current_batch_count < batch_leaf_size &&
           current_iter_ < config_->num_mcts_iteration) {
      MCTSNode* leaf = Select();

      Expand(leaf);

      // If the leaf node is already computed, then no need to use the virtual
      // loss.
      if (leaf->Computed()) {
        // Evaluate the current position from the perspective of the current
        // player of leaf. If it is good, then it means it is bad for the
        // previous player. So when we backpropagate, we alternate the sign of
        // q.
        float q = Evaluate(leaf);
        leaf->SetValueOfThisState(q);

        Backup(leaf);
      } else {
        // If the node is not computed yet, then we add to the batch_nodes and
        // specify the virtual loss instead.
        BackupVirtual(leaf, config_->mcts_virtual_loss);
        batch_leaf_nodes.push_back(leaf);
        current_batch_count++;
      }

      current_iter_++;
    }

    if (!batch_leaf_nodes.empty()) {
      std::vector<const GameState*> states;
      states.reserve(batch_leaf_nodes.size());

      for (MCTSNode* leaf_node : batch_leaf_nodes) {
        states.push_back(&leaf_node->State());
      }

      std::vector<float> q_s;
      if (config_->use_async_inference) {
        q_s = evaluator_->EvaluateAsyncBatch(states, worker_id_);
      } else {
        q_s = evaluator_->EvalulateBatch(states);
      }

      for (size_t i = 0; i < q_s.size(); i++) {
        batch_leaf_nodes[i]->SetValueOfThisState(q_s[i]);
        Backup(batch_leaf_nodes[i]);
        ClearVirtual(batch_leaf_nodes[i]);
      }
    }
  }
}

void MCTS::DoSingleRun() {
  for (; current_iter_ < config_->num_mcts_iteration; current_iter_++) {
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
  std::vector<float> dist = dist_->GetDistribution(possible_moves.size());

  for (size_t i = 0; i < possible_moves.size(); i++) {
    const Move& move = possible_moves[i];
    float noise = dist[i];

    nodes_.push_back(
        std::make_unique<MCTSNode>(std::make_unique<GameState>(&state, move),
                                   node, ComputePrior(node->Prior(), noise)));
    node->AddChildNode(nodes_.back().get(), move);
  }

  // Shuffle the ordering of the child node visit (for the randomization).
  std::shuffle(node->Children().begin(), node->Children().end(),
               config_->rand_gen);

  // For the root node, evey child will be visited anyway. So we just batch run
  // every nodes.
  if (root_ == node) {
    std::vector<std::vector<const GameState*>> batches = CreateBatches(
        node, config_->mcts_inference_batch_size, possible_moves.size());
    PreComputeBatches(node, evaluator_, batches);
  } else if (node->Parent()->Visit() >=
             config_->precompute_batch_parent_min_visit_count) {
    // If the parent was visited more than 2 times before, then it is likely
    // that every child node of this parent will get visited too. Hence let's
    // just precompute all the values of child.
    std::vector<std::vector<const GameState*>> batches =
        CreateBatches(node->Parent(), config_->mcts_inference_batch_size,
                      config_->mcts_inference_batch_size);
    PreComputeBatches(node->Parent(), evaluator_, batches);
  }
}

float MCTS::Evaluate(const MCTSNode* node) {
  if (node->Computed()) {
    return node->V();
  }

  if (config_->use_async_inference) {
    return evaluator_->EvaluateAsync(node->State(), worker_id_);
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

void MCTS::ClearVirtual(MCTSNode* leaf_node) {
  MCTSNode* current = leaf_node;
  while (current) {
    current->ClearVirtualLoss();
    current = current->Parent();
  }
}

void MCTS::BackupVirtual(MCTSNode* leaf_node, float virtual_loss) {
  // Virtual loss is added so that the same path is not visited again during the
  // batch MCTS.
  MCTSNode* current = leaf_node;

  while (current) {
    current->AddVirtualLoss(virtual_loss);
    current = current->Parent();
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

Move MCTS::MoveToMake(bool choose_best_move) const {
  if (choose_best_move) {
    std::optional<Move> best_move;

    int max_visit = 0;
    float max_value = -1000;

    for (const auto& [child_node, move] : root_->Children()) {
      if (child_node->Visit() == max_visit) {
        if (child_node->Q() >= max_value) {
          best_move = move;
          max_value = child_node->Q();
        }
      } else if (child_node->Visit() > max_visit) {
        max_visit = child_node->Visit();
        best_move = move;
      }
    }

    return best_move.value();
  }

  int current_count = 0;
  std::vector<std::pair<Move, int>> move_and_cumulative_count;
  for (const auto& [child_node, move] : root_->Children()) {
    current_count += child_node->Visit();
    move_and_cumulative_count.push_back(std::make_pair(move, current_count));
  }

  std::uniform_int_distribution<> distrib(0, current_count - 1);
  int rand_num = distrib(config_->rand_gen);
  for (const auto& [move, cumulative] : move_and_cumulative_count) {
    if (rand_num < cumulative) {
      return move;
    }
  }

  // This should not happen.
  assert(false);
  return move_and_cumulative_count[0].first;
}

void MCTS::ShowPath(MCTSNode* node) const {
  std::vector<MCTSNode*> path;

  MCTSNode* current = node;
  while (current) {
    path.push_back(current);
    current = current->Parent();
  }

  std::vector<std::string> moves;
  for (int i = path.size() - 1; i >= 1; i--) {
    // Find a path from path[i] to path[i - 1];
    const auto& children = path[i]->Children();
    for (const auto& [node, move] : children) {
      if (node == path[i - 1]) {
        auto s = fmt::format("{} (Q {} N {} VL {})", move.Str(), node->Q(),
                             node->Visit(), node->VirtualLoss());
        moves.push_back(s);
      }
    }
  }

  fmt::print("[Worker {}] {} \n", worker_id_, absl::StrJoin(moves, " -> "));
}

void MCTS::DumpDebugInfo() const { DumpDebugInfo(root_, 0); }

void MCTS::DumpDebugInfo(MCTSNode* node, int depth) const {
  if (node->Visit() == 0) {
    return;
  }

  for (int i = 0; i < depth - 1; i++) {
    fmt::print(" ");
  }

  if (node == root_) {
    fmt::print("{:02} Root {} ", depth, node->Q());
  } else {
    fmt::print("{:02} {} {} ", depth, node->State().LastMove().Str(),
               node->Q());
  }

  node->DumpDebugInfo();
  for (const auto& n : node->Children()) {
    DumpDebugInfo(n.first, depth + 1);
  }

  // Best moves.
  if (node == root_) {
    std::vector<std::pair<MCTSNode*, Move>> children = root_->Children();
    if (children.empty()) {
      return;
    }

    for (auto [node, m] : children) {
      fmt::print("{} : ", m.Str());
      node->DumpDebugInfo();
    }
  }
}

}  // namespace chess
