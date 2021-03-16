#ifndef TRAIN_H
#define TRAIN_H

#include "agent.h"
#include "device.h"
#include "nn/chess_nn.h"

namespace chess {

class Train {
 public:
  Train(DeviceManager* device_manager)
      : device_manager_(device_manager), current_best_(/*num_layer=*/10) {
    current_best_->to(device_manager_->Device());
  }

  void DoTrain(int num_threads);

 private:
  void GenerateExperience();

  // Train the train target.
  void TrainNN(ChessNN train_target);

  bool IsTrainedBetter(ChessNN train_target);

  DeviceManager* device_manager_;
  ChessNN current_best_;

  std::mutex exp_guard_;
  std::vector<std::unique_ptr<Experience>> experiences_;

  std::atomic<int> total_exp_ = 0;
};

}  // namespace chess

#endif
