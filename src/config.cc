#include "config.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace chess {

using json = nlohmann::json;

template <typename T>
std::string to_string_wrapper(const T& v) {
  if constexpr (std::is_same_v<T, std::string>) {
    return v;
  } else {
    return std::to_string(v);
  }
}

#define DEFINE_CONFIG(config, type)            \
  if (config_data.count(#config)) {            \
    config = config_data[#config].get<type>(); \
  }                                            \
  config_str_ += std::string(#config) + ":" + to_string_wrapper(config) + "\n";

Config::Config(std::string file_name) {
  std::ifstream in(file_name.c_str());

  json config_data;
  in >> config_data;

  DEFINE_CONFIG(num_threads, int);
  DEFINE_CONFIG(num_layer, int);
  DEFINE_CONFIG(num_epoch, int);
  DEFINE_CONFIG(total_game_play_for_testing, int);
  DEFINE_CONFIG(current_best_target_score, float);
  DEFINE_CONFIG(max_game_moves_until_draw, int);
  DEFINE_CONFIG(num_mcts_iteration, int);
  DEFINE_CONFIG(mcts_inference_batch_size, int);
  DEFINE_CONFIG(do_batch_mcts , bool);
  DEFINE_CONFIG(mcts_batch_leaf_node_size, int);
  DEFINE_CONFIG(mcts_virtual_loss, float);
  DEFINE_CONFIG(train_batch_size, int);
  DEFINE_CONFIG(num_self_play_game, int);
  DEFINE_CONFIG(learning_rate, float);
  DEFINE_CONFIG(weight_decay, float);
  DEFINE_CONFIG(existing_model_name, std::string);
  DEFINE_CONFIG(dirichlet_noise, float);
  DEFINE_CONFIG(show_self_play_boards, bool);
  DEFINE_CONFIG(use_async_inference, bool);
  DEFINE_CONFIG(precompute_batch_parent_min_visit_count, int);
  DEFINE_CONFIG(run_server, bool);
  DEFINE_CONFIG(server_port, std::string);
  DEFINE_CONFIG(use_cuda, bool);

  if (!use_cuda) {
    device = torch::Device(torch::kCPU);
  }
}

void Config::PrintConfig() const { std::cout << config_str_ << std::endl; }

}  // namespace chess
