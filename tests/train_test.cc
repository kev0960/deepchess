#include "train.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nn/nn_util.h"
#include "test_utils.h"

namespace chess {
namespace {

std::unique_ptr<Experience> CreateExperience(
    std::unique_ptr<GameState> state, std::vector<std::pair<Move, float>> move,
    float reward) {
  auto policy = MoveToTensor(move).unsqueeze(0).flatten(1);

  return std::make_unique<Experience>(std::move(state), policy, reward);
}

TEST(TrainTest, TrainNN) {
  Config config;
  Train trainer(&config);

  GameStateBuilder builder;
  builder
      .DoMove(Move(7, 1, 5, 2))  // Nb3
      .DoMove(Move(0, 1, 2, 2))  // Nc6
      /* White Knight goes back to original position. */
      .DoMove(Move(5, 2, 7, 1))
      /* Black Knight goes back to original position. */
      .DoMove(Move(2, 2, 0, 1));

  trainer.AddExperienceForTesting(
      CreateExperience(builder.ReleaseStateAt(0),
                       {{Move(7, 1, 5, 2), 0.9}, {Move(6, 1, 4, 1), 0.1}}, 1));
  trainer.AddExperienceForTesting(
      CreateExperience(builder.ReleaseStateAt(1),
                       {{Move(0, 1, 2, 2), 0.6}, {Move(1, 3, 3, 3), 0.4}}, -1));
  trainer.AddExperienceForTesting(
      CreateExperience(builder.ReleaseStateAt(2),
                       {{Move(5, 2, 7, 1), 0.8}, {Move(6, 3, 4, 3), 0.1}}, 1));
  trainer.AddExperienceForTesting(CreateExperience(builder.ReleaseStateAt(3),
                                                   {{Move(2, 2, 0, 1), 0.55},
                                                    {Move(1, 5, 3, 5), 0.4},
                                                    {Move(1, 7, 3, 7), 0.05}},
                                                   -1));
  trainer.TrainNN();
}

TEST(TrainTest, AgentPlayTest) {
  Config config;
  config.num_mcts_iteration = 2;
  config.total_game_play_for_testing = 20;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;
  config.show_self_play_boards = false;

  Train trainer(&config);

  ChessNN nn(10);
  nn->to(config.device);

  Evaluator target_eval(nn, &config);
  target_eval.StartInferenceWorker();

  Evaluator current_eval(nn, &config);
  current_eval.StartInferenceWorker();

  // Every match should be draw.
  EXPECT_TRUE(trainer.IsTrainedBetter(&target_eval, &current_eval));
}

/*
TEST(TrainTest, BenchmarkTimeUsingAsync) {
  Config config;
  config.num_threads = 32;
  config.num_mcts_iteration = 600;
  config.use_async_inference = true;
  config.total_game_play_for_testing = 32;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;
  config.show_self_play_boards = false;

  config.precompute_batch_parent_min_visit_count = 3;

  Train trainer(&config);

  auto start = std::chrono::high_resolution_clock::now();

  // Every match should be draw.
  EXPECT_TRUE(trainer.IsTrainedBetter());

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time for Entire Game"
            << ms.count() / 1000.0 / config.num_threads << "ms" << std::endl;
}

TEST(TrainTest, BenchmarkTimeUsingPreCompute) {
  Config config;
  config.num_threads = 32;
  config.num_mcts_iteration = 600;
  config.use_async_inference = false;
  config.total_game_play_for_testing = 32;
  config.max_game_moves_until_draw = 10;
  config.current_best_target_score = 10;
  config.precompute_batch_parent_min_visit_count = 3;
  config.show_self_play_boards = false;

  Train trainer(&config);

  auto start = std::chrono::high_resolution_clock::now();

  // Every match should be draw.
  EXPECT_TRUE(trainer.IsTrainedBetter());

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time for Entire Game"
            << ms.count() / 1000.0 / config.num_threads << "ms" << std::endl;
}
*/

TEST(TrainTest, BenchmarkTrainTimeUsingAsyncInference) {
  Config config;
  config.num_epoch = 1;
  config.num_threads = 32;
  config.num_self_play_game = 32;
  config.num_mcts_iteration = 600;
  config.mcts_inference_batch_size = 64;
  config.use_async_inference = true;
  config.total_game_play_for_testing = 50;
  config.max_game_moves_until_draw = 20;
  config.show_self_play_boards = false;
  config.precompute_batch_parent_min_visit_count = 3;
  config.existing_model_name = "SomeModelForTesting.pt";

  Train trainer(&config);

  auto start = std::chrono::high_resolution_clock::now();

  trainer.DoTrain();

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time for Entire Train" << ms.count() / 1000.0 << "s"
            << std::endl;
}

/*
TEST(TrainTest, BenchmarkTrainTimeUsingPrecomputeOnly) {
  Config config;
  config.num_epoch = 1;
  config.num_threads = 12;
  config.num_self_play_game = 32;
  config.num_mcts_iteration = 600;
  config.use_async_inference = false;
  config.total_game_play_for_testing = 32;
  config.max_game_moves_until_draw = 10;
  config.show_self_play_boards = false;
  config.existing_model_name = "SomeModelForTesting.pt";

  Train trainer(&config);

  auto start = std::chrono::high_resolution_clock::now();

  trainer.DoTrain();

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time for Entire Train" << ms.count() / 1000.0 << "s"
            << std::endl;
}
*/

}  // namespace
}  // namespace chess
