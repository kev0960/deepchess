#include "evaluator.h"

#include "nn/chess_nn.h"
#include "nn/nn_util.h"

namespace chess {

float Evaluator::Evalulate(const GameState& state) {
  if (state.IsDraw()) {
    return 0;
  }

  if (state.GetLegalMoves().empty()) {
    // If it is a checkmate, then it is done :(
    return -1;
  }

  torch::Tensor tensor = GameStateToTensor(state);
  tensor = tensor.to(config_->device);

  // Convert board to the state.
  torch::Tensor value_tensor = chess_net_->GetValue(tensor);

  // Note that the returned value_tensor is 1 * 1.
  torch::Device device(torch::kCPU);
  torch::Tensor cpu_tensor = value_tensor.to(device);

  return cpu_tensor.data_ptr<float>()[0];
}

std::vector<float> Evaluator::EvalulateBatch(
    std::vector<const GameState*> states) {
  if (states.empty()) {
    return {};
  }

  std::vector<float> scores(states.size(), 0);

  std::vector<bool> is_set(states.size(), false);
  std::vector<torch::Tensor> batch;
  for (size_t i = 0; i < states.size(); i++) {
    if (states[i]->IsDraw()) {
      scores[i] = 0;
      is_set[i] = true;
      continue;
    }

    if (states[i]->GetLegalMoves().empty()) {
      scores[i] = -1;
      is_set[i] = true;
      continue;
    }

    torch::Tensor tensor = GameStateToTensor(*states[i]);
    batch.push_back(tensor);
  }

  if (batch.empty()) {
    return scores;
  }

  torch::Tensor batch_tensor = torch::stack(batch);
  batch_tensor = batch_tensor.to(config_->device);

  torch::Tensor value_tensor = chess_net_->GetValue(batch_tensor);

  // Note that the returned value_tensor is N * 1.
  torch::Device device(torch::kCPU);
  torch::Tensor cpu_tensor = value_tensor.to(device);

  size_t score_index = 0, batch_index = 0;
  while (batch_index < batch.size()) {
    if (is_set[score_index]) {
      score_index++;
      continue;
    } else {
      scores[score_index] = cpu_tensor.data_ptr<float>()[batch_index];
    }

    score_index++;
    batch_index++;
  }

  return scores;
}

float Evaluator::EvaluateAsync(const GameState& state, int worker_id) {
  if (state.IsDraw()) {
    return 0;
  }

  if (state.GetLegalMoves().empty()) {
    // If it is a checkmate, then it is done :(
    return -1;
  }

  std::unique_lock<std::mutex> lk(worker_info_[worker_id].m_cv);

  {
    std::lock_guard<std::mutex> lk_queue(batch_queue_m_);
    batch_queue_.push_back(std::make_pair(GameStateToTensor(state), worker_id));
  }
  batch_queue_cv_.notify_one();

  // Wait until the inference is done.
  worker_info_[worker_id].cv_inference.wait(lk);

  return worker_info_[worker_id].result;
}

void Evaluator::InferenceWorker() {
  while (!should_finish_inference_) {
    std::unique_lock<std::mutex> lk(batch_queue_m_);
    batch_queue_cv_.wait(lk, [this]() {
      return !batch_queue_.empty() || should_finish_inference_;
    });

    std::vector<torch::Tensor> batches;
    std::vector<int> workers;

    batches.reserve(batch_queue_.size());
    workers.reserve(batch_queue_.size());
    for (auto [tensor, worker_id] : batch_queue_) {
      batches.push_back(tensor);
      workers.push_back(worker_id);
    }

    batch_queue_.clear();
    lk.unlock();

    if (batches.empty()) {
      continue;
    }

    torch::Tensor batch_tensor = torch::stack(batches).to(config_->device);
    torch::Tensor value_tensor = chess_net_->GetValue(batch_tensor);

    torch::Device device(torch::kCPU);
    torch::Tensor cpu_tensor = value_tensor.to(device);
    for (size_t i = 0; i < workers.size(); i++) {
      worker_info_[workers[i]].result = cpu_tensor.data_ptr<float>()[i];
      worker_info_[workers[i]].cv_inference.notify_one();
    }
  }
}

void Evaluator::StartInferenceWorker() {
  inference_worker_ =
      std::make_unique<std::thread>([this]() { InferenceWorker(); });
}

Evaluator::~Evaluator() {
  should_finish_inference_ = true;
  if (inference_worker_) {
    batch_queue_cv_.notify_one();
    inference_worker_->join();
  }
}

}  // namespace chess

