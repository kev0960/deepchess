#include "train.h"

#include <random>

#include "agent.h"
#include "chess.h"
#include "nn/nn_util.h"

namespace chess {
namespace {

constexpr int kTotalGamePlayForTest = 100;

void ShuffleVector(std::vector<std::unique_ptr<Experience>>& exp) {
  auto rd = std::random_device();
  auto rng = std::default_random_engine(rd());

  std::shuffle(exp.begin(), exp.end(), rng);
}

}  // namespace

void Train::DoTrain(int num_threads) {
  (void)num_threads;
  torch::save(current_best_, "CurrentBest.pt");

  std::vector<std::thread> exp_generators;
  for (int i = 0; i < 8; i++) {
    exp_generators.push_back(std::thread(&Train::GenerateExperience, this));
  }

  for (auto& gen : exp_generators) {
    gen.join();
  }

  std::cout << "Num exps : " << experiences_.size() << std::endl;

  ChessNN train_target(10);
  torch::load(train_target, "CurrentBest.pt");
  train_target->to(device_manager_->Device());

  TrainNN(train_target);

  std::cout << "Is train better? " << IsTrainedBetter(train_target) << std::endl;
}

void Train::GenerateExperience() {
  ChessNN train_target(10);
  torch::load(train_target, "CurrentBest.pt");
  train_target->to(device_manager_->Device());

  DirichletDistribution dirichlet(0.3);

  while (total_exp_ < 8) {
    total_exp_++;

    Agent agent(train_target, &dirichlet, device_manager_);
    agent.Run();

    auto& experiences = agent.GetExperience();

    exp_guard_.lock();
    experiences_.insert(experiences_.end(),
                        std::make_move_iterator(experiences.begin()),
                        std::make_move_iterator(experiences.end()));
    exp_guard_.unlock();
  }
}

void Train::TrainNN(ChessNN train_target) {
  ShuffleVector(experiences_);

  torch::optim::Adam optimizer(
      train_target->parameters(),
      torch::optim::AdamOptions(2e-4).weight_decay(1e-4));
  for (const auto& exp : experiences_) {
    torch::Tensor state_tensor = GameStateToTensor(*exp->state);
    state_tensor = state_tensor.to(device_manager_->Device());

    train_target->zero_grad();

    torch::Tensor value = train_target->GetValue(state_tensor);

    // We have to zero-out the probability for impossible actions. 
    torch::Tensor policy =
        NormalizePolicy(*exp->state, train_target->GetPolicy(state_tensor));

    torch::Tensor total_loss =
        torch::binary_cross_entropy(policy, exp->policy) +
        torch::norm(value - exp->result);

    total_loss.backward();
    optimizer.step();

    std::cout << "Total loss : " << total_loss << std::endl;
  }

  if (IsTrainedBetter(train_target)) {
    current_best_ = train_target;
  }
}

bool Train::IsTrainedBetter(ChessNN train_target) {
  DirichletDistribution no_noise(0);

  Agent target(train_target, &no_noise, device_manager_);
  Agent current(current_best_, &no_noise, device_manager_);

  Chess chess;

  // Now play game each other.
  float target_score = 0;  // Win 1, Draw 0.5, Lose 0.
  for (int i = 0; i < kTotalGamePlayForTest; i++) {
    auto result = chess.PlayChessBetweenAgents(&target, &current);
    if (result == DRAW) {
      target_score += 0.5;
    } else if (result == WHITE_WIN) {
      target_score += 1;
    }
  }

  for (int i = 0; i < kTotalGamePlayForTest; i++) {
    auto result = chess.PlayChessBetweenAgents(&current, &target);
    if (result == DRAW) {
      target_score += 0.5;
    } else if (result == BLACK_WIN) {
      target_score += 1;
    }
  }

  std::cout << "Score : " << target_score << std::endl;

  if (target_score >= 55) {
    return true;
  }

  return false;
}

}  // namespace chess
