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

  Train trainer(&config);

  // Every match should be draw.
  EXPECT_TRUE(trainer.IsTrainedBetter());
}

}  // namespace
}  // namespace chess
