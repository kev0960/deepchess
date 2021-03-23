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
  DirichletDistribution dist(0);

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

  DirichletDistribution dist(0);

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
