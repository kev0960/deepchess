#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include <future>

#include "config.h"
#include "game_state.h"
#include "nn/chess_nn.h"

namespace chess {

struct EvaluatorWorkerInfo {
  // Condition variable to wait for the inference to finish.
  std::condition_variable cv_inference;

  // Mutex for cv.
  std::mutex m_cv;

  // Hold the result of the inference.
  float result;
};

class Evaluator {
 public:
  Evaluator(ChessNN chess_net, const Config* config)
      : chess_net_(chess_net),
        config_(config),
        worker_info_(config_->num_threads) {}

  virtual float Evalulate(const GameState& board);
  virtual std::vector<float> EvalulateBatch(
      std::vector<const GameState*> boards);

  // When used, every other EvaluateAsync that are fired at the similar
  // time will be batched together.
  virtual float EvaluateAsync(const GameState& state, int worker_id);

  void InferenceWorker();
  void StartInferenceWorker();

  // Join inference worker.
  ~Evaluator();

 private:
  ChessNN chess_net_;
  const Config* config_;

  std::mutex batch_queue_m_;
  std::condition_variable batch_queue_cv_;
  std::deque<std::pair<torch::Tensor, int>> batch_queue_;

  std::vector<EvaluatorWorkerInfo> worker_info_;

  bool should_finish_inference_ = false;
  std::unique_ptr<std::thread> inference_worker_;
};

}  // namespace chess

#endif
