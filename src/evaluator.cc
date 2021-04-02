#include "evaluator.h"

#include <fmt/ranges.h>

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

  auto& worker_info = worker_info_[worker_id];

  std::unique_lock<std::mutex> lk(worker_info.m_cv);
  worker_info.result_is_set = false;

  {
    std::lock_guard<std::mutex> lk_queue(batch_queue_m_);
    torch::Tensor tensor = GameStateToTensor(state);
    assert(tensor.sizes().size() == 3);

    // We have to pass 1 * .... tensor;
    tensor = tensor.unsqueeze(0);
    batch_queue_.push_back(std::make_pair(tensor, worker_id));
  }

  batch_queue_cv_.notify_one();

  // Wait until the inference is done.
  worker_info_[worker_id].cv_inference.wait(
      lk, [&worker_info]() { return worker_info.result_is_set; });

  assert(worker_info.result.size() == 1);
  return worker_info.result[0];
}

std::vector<float> Evaluator::EvaluateAsyncBatch(
    const std::vector<const GameState*>& states, int worker_id) {
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

  auto& worker_info = worker_info_[worker_id];

  std::unique_lock<std::mutex> lk(worker_info.m_cv);
  worker_info.result_is_set = false;

  {
    std::lock_guard<std::mutex> lk_queue(batch_queue_m_);
    batch_queue_.push_back(std::make_pair(batch_tensor, worker_id));
  }

  batch_queue_cv_.notify_one();

  // Wait until the inference is done.
  worker_info_[worker_id].cv_inference.wait(
      lk, [&worker_info]() { return worker_info.result_is_set; });

  size_t score_index = 0, batch_index = 0;
  while (score_index < states.size()) {
    if (is_set[score_index]) {
      score_index++;
      continue;
    }

    scores[score_index] = worker_info.result[batch_index];
    score_index++;
    batch_index++;
  }

  return scores;
}

void Evaluator::InferenceWorker() {
  while (!should_finish_inference_) {
    std::unique_lock<std::mutex> lk(batch_queue_m_);
    batch_queue_cv_.wait(lk, [this]() {
      return !batch_queue_.empty() || should_finish_inference_;
    });

    std::vector<torch::Tensor> batches;
    std::vector<std::pair</*worker_id=*/int, /*batch_size=*/int>> workers;

    batches.reserve(batch_queue_.size());
    workers.reserve(batch_queue_.size());
    for (auto [tensor, worker_id] : batch_queue_) {
      batches.push_back(tensor);
      workers.push_back(std::make_pair(worker_id, tensor.sizes()[0]));
    }

    batch_queue_.clear();
    lk.unlock();

    if (batches.empty()) {
      continue;
    }

    torch::Tensor batch_tensor = torch::cat(batches).to(config_->device);
    assert(batch_tensor.sizes()[1] == 119);

    torch::Tensor value_tensor = chess_net_->GetValue(batch_tensor);

    torch::Device device(torch::kCPU);
    torch::Tensor cpu_tensor = value_tensor.to(device);

    int batch_index = 0;
    for (size_t worker_index = 0; worker_index < workers.size();
         worker_index++) {
      auto& worker_info = worker_info_[workers[worker_index].first];
      worker_info.result.clear();

      const int worker_batch_size = workers[worker_index].second;
      worker_info.result.reserve(worker_batch_size);

      for (int i = 0; i < worker_batch_size; i++) {
        worker_info.result.push_back(cpu_tensor.data_ptr<float>()[batch_index]);
        batch_index++;
      }

      worker_info.result_is_set = true;
      worker_info.cv_inference.notify_one();
    }

    assert(batch_index == cpu_tensor.sizes()[0]);
  }
}

void Evaluator::StartInferenceWorker() {
  // TODO Does multiple inference worker work?
  for (int i = 0; i < config_->evaluator_worker_count; i++) {
    inference_workers_.push_back(
        std::thread(&Evaluator::InferenceWorker, this));
  }
}

Evaluator::~Evaluator() {
  should_finish_inference_ = true;
  if (!inference_workers_.empty()) {
    std::cerr << "Deleting evaluators.." << std::endl;
    batch_queue_cv_.notify_all();

    for (auto& worker : inference_workers_) {
      worker.join();
    }
  }
}

}  // namespace chess

