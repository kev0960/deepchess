#ifndef AGENT_H
#define AGENT_H

#include <torch/torch.h>

#include <vector>

#include "config.h"
#include "distribution.h"
#include "evaluator.h"
#include "game_state.h"
#include "nn/chess_nn.h"
#include "worker_manager.h"

namespace chess {

struct Experience {
  std::unique_ptr<GameState> state;
  torch::Tensor policy;

  // If the one who plays at this state wins, it should be 1. If lost, then -1.
  // Draw is 0.
  float result = 0;

  Experience(std::unique_ptr<GameState> state, torch::Tensor policy,
             float result)
      : state(std::move(state)), policy(policy), result(result) {}
};

class Agent {
 public:
  Agent(Distribution* dist, Config* config, Evaluator* evaluator,
        WorkerManager* worker_manager, int worker_id);

  // Conduct the self play and gain experiences.
  void Run();

  // Not const since it has to be shuffled.
  std::vector<std::unique_ptr<Experience>>& GetExperience() {
    return experiences_;
  }

  Move GetBestMove(const GameState& game_state) const;
  int WorkerId() const { return worker_id_; }

 private:
  void DoSelfPlay();

  std::vector<std::unique_ptr<Experience>> experiences_;

  Distribution* dist_;
  Config* config_;
  Evaluator* evaluator_;
  WorkerManager* worker_manager_;

  // ID of the current thread worker.
  int worker_id_;
};

}  // namespace chess

#endif
