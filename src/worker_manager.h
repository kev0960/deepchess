#ifndef WORKER_MANAGER_H
#define WORKER_MANAGER_H

#include "config.h"

namespace chess {

struct TrainWorkerInfo {
  // Number of moves played in the current game.
  int current_game_total_move = 0;

  // Total # of games played.
  int total_game_played = 0;
};

struct InferenceWorkerInfo {
  // # of inferences done.
  uint64_t total_num_inference = 0;

  // Total # of states that are inferenced.
  uint64_t total_inference_batch_size = 0;
};

class WorkerManager {
 public:
  WorkerManager(Config* config)
      : config_(config),
        train_worker_info_(config_->num_threads),
        inference_worker_info_(config_->evaluator_worker_count) {}

  TrainWorkerInfo& GetWorkerInfo(int worker_id) {
    return train_worker_info_[worker_id];
  }

  void ResetWorkerInfo() {
    train_worker_info_.clear();
    train_worker_info_.resize(config_->num_threads);

    inference_worker_info_.clear();
    inference_worker_info_.resize(config_->evaluator_worker_count);
  }

  InferenceWorkerInfo& GetInferenceWorkerInfo(int worker_id) {
    return inference_worker_info_[worker_id];
  }

 private:
  Config* config_;

  std::vector<TrainWorkerInfo> train_worker_info_;
  std::vector<InferenceWorkerInfo> inference_worker_info_;
};

}  // namespace chess

#endif
