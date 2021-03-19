#include "train.h"

#include <random>

#include "agent.h"
#include "chess.h"
#include "nn/nn_util.h"

namespace chess {
namespace {

void ShuffleVector(std::vector<std::unique_ptr<Experience>>& exp) {
  auto rd = std::random_device();
  auto rng = std::default_random_engine(rd());

  std::shuffle(exp.begin(), exp.end(), rng);
}

std::vector<std::vector<const Experience*>> CreateExperienceBatch(
    const std::vector<std::unique_ptr<Experience>>& experiences,
    size_t batch_size) {
  std::vector<std::vector<const Experience*>> batches;

  for (size_t i = 0; i < experiences.size(); i++) {
    if (batches.empty() || batches.back().size() >= batch_size) {
      batches.emplace_back();
    }

    std::vector<const Experience*>& batch = batches.back();
    batch.push_back(experiences.at(i).get());
  }

  return batches;
}

}  // namespace

void Train::DoTrain() {
  for (int i = 0; i < config_->num_epoch; i++) {
    torch::save(current_best_, "CurrentBest.pt");
    torch::load(train_target_, "CurrentBest.pt");
    train_target_->to(config_->device);

    std::vector<std::thread> exp_generators;
    for (int i = 0; i < config_->num_threads; i++) {
      exp_generators.push_back(std::thread(&Train::GenerateExperience, this));
    }

    for (auto& gen : exp_generators) {
      gen.join();
    }

    std::cout << "Num exps : " << experiences_.size() << std::endl;

    TrainNN();

    if (IsTrainedBetter()) {
      current_best_ = train_target_;
    }
  }
}

void Train::GenerateExperience() {
  DirichletDistribution dirichlet(0.3);

  while (total_exp_ < config_->num_self_play_game) {
    total_exp_++;

    Agent agent(train_target_, &dirichlet, config_);
    agent.Run();

    auto& experiences = agent.GetExperience();

    exp_guard_.lock();
    experiences_.insert(experiences_.end(),
                        std::make_move_iterator(experiences.begin()),
                        std::make_move_iterator(experiences.end()));
    exp_guard_.unlock();
  }
}

void Train::TrainNN() {
  ShuffleVector(experiences_);

  torch::optim::Adam optimizer(train_target_->parameters(),
                               torch::optim::AdamOptions(config_->learning_rate)
                                   .weight_decay(config_->weight_decay));

  size_t total = experiences_.size();
  size_t done = 0;

  std::vector<std::vector<const Experience*>> batches =
      CreateExperienceBatch(experiences_, config_->train_batch_size);

  for (const auto& batch : batches) {
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> target_policies;
    std::vector<torch::Tensor> input_policies;
    std::vector<float> results;

    for (const Experience* exp : batch) {
      torch::Tensor state_tensor =
          GameStateToTensor(*exp->state.get()).to(config_->device);
      states.push_back(state_tensor);

      target_policies.push_back(exp->policy.to(config_->device));
      results.push_back(exp->result);

      input_policies.push_back(
          NormalizePolicy(*exp->state, train_target_->GetPolicy(state_tensor)));
    }

    torch::Tensor state_batch = torch::stack(states).to(config_->device);

    train_target_->zero_grad();

    torch::Tensor input_values =
        train_target_->GetValue(state_batch).to(config_->device);
    torch::Tensor target_values =
        torch::from_blob(results.data(), {(long)batch.size(), 1})
            .to(config_->device);

    torch::Tensor input_policy = torch::stack(input_policies).flatten(0);
    torch::Tensor target_policy = torch::stack(target_policies).flatten(0);

    torch::Tensor total_loss =
        -torch::dot(torch::log(target_policy).clamp(-1000), input_policy) +
        torch::norm(input_values - target_values);

    total_loss.backward();
    optimizer.step();

    done += batch.size();

    std::cout << "Loss[" << done << " / " << total
              << "] : " << total_loss.item<float>() << std::endl;
  }
}

bool Train::IsTrainedBetter() {
  std::vector<std::thread> agent_evaluators;
  for (int i = 0; i < config_->num_threads; i++) {
    agent_evaluators.push_back(std::thread(&Train::PlayGamesEachOther, this));
  }

  for (auto& eval : agent_evaluators) {
    eval.join();
  }

  if (target_score_ >= config_->current_best_target_score) {
    return true;
  }

  return false;
}

void Train::PlayGamesEachOther() {
  DirichletDistribution no_noise(0);

  Agent target(train_target_, &no_noise, config_);
  Agent current(current_best_, &no_noise, config_);

  Chess chess;

  while (current_game_playing_ < config_->total_game_play_for_testing) {
    int current_game_index = current_game_playing_.fetch_add(1);
    if (current_game_index < config_->total_game_play_for_testing / 2) {
      auto result = chess.PlayChessBetweenAgents(
          &target, &current, config_->max_game_moves_until_draw);
      if (result == DRAW) {
        fmt::print("Target [White] Draw \n");
        target_score_ += 1;
      } else if (result == WHITE_WIN) {
        fmt::print("Target [White] Win \n");
        target_score_ += 2;
      }
    } else {
      auto result = chess.PlayChessBetweenAgents(
          &current, &target, config_->max_game_moves_until_draw);
      if (result == DRAW) {
        fmt::print("Target [Black] Draw \n");
        target_score_ += 1;
      } else if (result == BLACK_WIN) {
        fmt::print("Target [Black] Win \n");
        target_score_ += 2;
      }
    }
  }
}

void Train::AddExperienceForTesting(std::unique_ptr<Experience> exp) {
  experiences_.push_back(std::move(exp));
}
}  // namespace chess
