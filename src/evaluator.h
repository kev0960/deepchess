#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <torch/torch.h>

#include <future>

#include "config.h"
#include "game_state.h"
#include "nn/chess_nn.h"
#include "worker_manager.h"

namespace chess {

struct EvaluatorWorkerInfo {
  // Condition variable to wait for the inference to finish.
  std::condition_variable cv_inference;

  // Mutex for cv.
  std::mutex m_cv;

  // True if the result is set.
  bool result_is_set = false;

  // Hold the result of the inference.
  std::vector<float> result;
};

class Evaluator {
 public:
  Evaluator(ChessNN chess_net, const Config* config,
            WorkerManager* worker_manager)
      : chess_net_(chess_net),
        config_(config),
        worker_info_(config_->num_threads),
        worker_manager_(worker_manager) {}

  virtual float Evalulate(const GameState& board);
  virtual std::vector<float> EvalulateBatch(
      std::vector<const GameState*> boards);

  // When used, every other EvaluateAsync that are fired at the similar
  // time will be batched together.
  virtual float EvaluateAsync(const GameState& state, int worker_id);
  virtual std::vector<float> EvaluateAsyncBatch(
      const std::vector<const GameState*>& states, int worker_id);

  void InferenceWorker(int worker_id);
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
  std::vector<std::thread> inference_workers_;

  WorkerManager* worker_manager_;
};

}  // namespace chess

#endif
