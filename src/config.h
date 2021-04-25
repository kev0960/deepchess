#ifndef CONFIG_H
#define CONFIG_H

#include <torch/torch.h>

#include <random>
#include <string>

namespace chess {

class Config {
 public:
  Config() : rand_gen(std::random_device()()) {}

  Config(std::string file_name);

  // # of worker threads that generate the experience.
  int num_threads = 8;

  // # of layers in the ChessNN.
  int num_layer = 10;

  // # of filters in ChessNN.
  int num_filter = 128;

  // Total number of entire training iteration.
  int num_epoch = 1;

  // Total # of games to play to see whether the trained network outperforms the
  // previous network.
  int total_game_play_for_testing = 100;

  // The minimum score that the target network should achieve when playing game
  // with the current best. Winning (2 pts), Draw (1 pt), Lose (0 pt)
  // Should be around 55% (total_game_play_for_testing * 1.1)
  int current_best_target_score = 110;

  // Maximum # of game moves until the self play ends with the draw.
  int max_game_moves_until_draw = 300;

  // # of MCTS iteration to make a decision.
  int num_mcts_iteration = 800;

  // Size of the batch for the MCTS roll-out inferencing.
  int mcts_inference_batch_size = 20;

  // Whether to do the batch MCTS.
  bool do_batch_mcts = false;

  // Size of the batch to request inferencing rollouts.
  int mcts_batch_leaf_node_size = 20;

  // Size of the virtual loss per visit.
  float mcts_virtual_loss = -0.05;

  // Size of the batch during training.
  int train_batch_size = 40;

  // # of self-play games.
  int num_self_play_game = 8;

  // Learning rate for Adam.
  float learning_rate = 0.01;

  // Weight decay for Adam.
  float weight_decay = 1e-4;

  // alpha of DirichletDistribution.
  float dirichlet_noise = 0.3;

  // Show boards of every move in the self-play.
  bool show_self_play_boards = true;

  // Show debug output of moves.
  bool move_debug_output = false;

  // Use the Async version of Inference evaluator.
  bool use_async_inference = false;

  // Minimum number of visit required to precompute value of every child of the
  // node. If this is set to num_mcts_iteration, then it means we won't
  // precompute batches.
  int precompute_batch_parent_min_visit_count = 2;

  // Number of async workers in Evaluator.
  int evaluator_worker_count = 1;
  
  // Model name to import.
  std::string existing_model_name = "";

  // Experience Save file.
  std::string exp_save_file_name = "";

  // Whether to use CUDA.
  bool use_cuda = true;
  torch::Device device = torch::Device(torch::kCUDA);

  // Should run the server.
  bool run_server = false;

  // Should do train.
  bool do_train = true;

  // Server port.
  std::string server_port = "8888";

  // Random number generator used across all.
  std::mt19937 rand_gen;

  void PrintConfig() const;

 private:
  std::string config_str_;
};

}  // namespace chess

#endif
