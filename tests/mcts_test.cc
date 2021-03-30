#include "mcts.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_utils.h"

namespace chess {
namespace {

class MCTSTest : public testing::Test {};

TEST_F(MCTSTest, CheckMoves) {
  Config config;
  config.num_mcts_iteration = 2;
  config.total_game_play_for_testing = 20;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;

  ChessNN nn(10);
  nn->to(config.device);

  Evaluator eval(nn, &config);
  UniformDistribution dist;

  GameStateBuilder builder;

  std::unordered_set<std::string> possible_moves;
  for (int i = 0; i < 10; i++) {
    MCTS mcts(builder.GetStates().front().get(), &eval, &dist, &config, 0);
    mcts.RunMCTS();
    possible_moves.insert(mcts.MoveToMake(/*choose_best_move=*/true).Str());
  }

  // Make sure that variety of moves are selected due to the random shuffling of
  // child nodes.
  EXPECT_TRUE(possible_moves.size() > 1);
}

TEST_F(MCTSTest, AsyncEval) {
  Config config;
  config.num_threads = 10;
  config.num_mcts_iteration = 100;
  config.total_game_play_for_testing = 20;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;
  config.use_async_inference = true;

  ChessNN nn(10);
  nn->to(config.device);

  Evaluator eval(nn, &config);
  eval.StartInferenceWorker();

  UniformDistribution dist;

  GameStateBuilder builder;

  std::vector<std::thread> workers;
  for (int i = 0; i < 10; i++) {
    workers.push_back(std::thread([&, i]() {
      MCTS mcts(builder.GetStates().front().get(), &eval, &dist, &config, i);
      mcts.RunMCTS();
    }));
  }

  for (auto& w : workers) {
    w.join();
  }
}

TEST_F(MCTSTest, BatchMCTSNotAsync) {
  Config config;
  config.num_threads = 10;
  config.num_mcts_iteration = 100;
  config.do_batch_mcts = true;
  config.mcts_batch_leaf_node_size = 20;

  config.total_game_play_for_testing = 20;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;
  config.use_async_inference = false;

  ChessNN nn(10);
  nn->to(config.device);

  Evaluator eval(nn, &config);
  eval.StartInferenceWorker();

  UniformDistribution dist;

  GameStateBuilder builder;

  std::vector<std::thread> workers;
  for (int i = 0; i < 10; i++) {
    workers.push_back(std::thread([&, i]() {
      MCTS mcts(builder.GetStates().front().get(), &eval, &dist, &config, i);
      mcts.RunMCTS();
    }));
  }

  for (auto& w : workers) {
    w.join();
  }
}

TEST_F(MCTSTest, BatchMCTSAsync) {
  Config config;
  config.num_threads = 10;
  config.num_mcts_iteration = 600;
  config.do_batch_mcts = true;
  config.mcts_batch_leaf_node_size = 20;

  config.total_game_play_for_testing = 20;
  config.max_game_moves_until_draw = 100;
  config.current_best_target_score = 10;
  config.use_async_inference = true;
  config.mcts_virtual_loss= -0.05;
  config.precompute_batch_parent_min_visit_count = 600;

  ChessNN nn(10);
  nn->to(config.device);

  Evaluator eval(nn, &config);
  eval.StartInferenceWorker();

  UniformDistribution dist;

  GameStateBuilder builder;

  std::vector<std::thread> workers;
  for (int i = 0; i < 10; i++) {
    workers.push_back(std::thread([&, i]() {
      MCTS mcts(builder.GetStates().front().get(), &eval, &dist, &config, i);
      mcts.RunMCTS();
    }));
  }

  for (auto& w : workers) {
    w.join();
  }
}

}  // namespace
}  // namespace chess
