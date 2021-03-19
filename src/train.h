#ifndef TRAIN_H
#define TRAIN_H

#include "agent.h"
#include "config.h"
#include "nn/chess_nn.h"

namespace chess {

class Train {
 public:
  Train(Config* config)
      : current_best_(config->num_layer),
        train_target_(config->num_layer),
        config_(config) {
    current_best_->to(config_->device);
    train_target_->to(config_->device);
  }

  void DoTrain();

  // Train the train target.
  void TrainNN();

  // Check whether the train_target performs better than current_best.
  bool IsTrainedBetter();

  void AddExperienceForTesting(std::unique_ptr<Experience> exp);

 private:
  void GenerateExperience();
  void PlayGamesEachOther();

  ChessNN current_best_;
  ChessNN train_target_;

  std::mutex exp_guard_;
  std::vector<std::unique_ptr<Experience>> experiences_;

  std::atomic<int> total_exp_ = 0;

  std::atomic<int> target_score_ = 0;

  // # of games that are being played between games now.
  std::atomic<int> current_game_playing_ = 0;

  Config* config_;
};

}  // namespace chess

#endif
