#ifndef TRAIN_H
#define TRAIN_H

#include "agent.h"
#include "config.h"
#include "nn/chess_nn.h"
#include "serialize.h"
#include "server_context.h"

namespace chess {

class Train {
 public:
  Train(Config* config, ServerContext* server_context)
      : current_best_(config->num_layer, config->num_filter),
        train_target_(config->num_layer, config->num_filter),
        config_(config),
        server_context_(server_context),
        experience_saver_(config) {
    current_best_->to(config_->device);
    train_target_->to(config_->device);
  }

  void DoTrain();

  // Train the train target.
  void TrainNN();

  // Check whether the train_target performs better than current_best.
  bool IsTrainedBetter(Evaluator* target_eval, Evaluator* current_eval);

  // Following two are exposed for the testing.
  void AddExperienceForTesting(std::unique_ptr<Experience> exp);
  ChessNN GetTrainTarget() { return train_target_; }

 private:
  void GenerateExperience(Evaluator* evaluator, int worker_id);
  void PlayGamesEachOther(Evaluator* target_eval, Evaluator* current_eval,
                          int worker_id);

  ChessNN current_best_;
  ChessNN train_target_;

  std::mutex exp_guard_;
  std::vector<std::unique_ptr<Experience>> experiences_;

  std::atomic<int> total_exp_ = 0;
  std::atomic<int> total_exp_done_ = 0;

  std::atomic<int> target_score_ = 0;

  // # of games that are being played between games now.
  std::atomic<int> current_game_playing_ = 0;

  Config* config_;
  ServerContext* server_context_;

  std::chrono::time_point<std::chrono::high_resolution_clock> exp_gen_start_;
  ExperienceSaver experience_saver_;
};

}  // namespace chess

#endif
