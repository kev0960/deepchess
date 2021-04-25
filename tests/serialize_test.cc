#include "serialize.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nn/nn_util.h"
#include "test_utils.h"

namespace chess {
namespace {

TEST(SerializeTest, Serialize) {
  GameStateBuilder builder;

  builder
      .DoMove(Move(7, 1, 5, 2))  // Nb3
      .DoMove(Move(0, 1, 2, 2))  // Nc6
      .DoMove(Move(6, 1, 4, 1))
      .DoMove(Move(1, 1, 3, 1));

  std::string save_file_name = "test_exp_save_file";

  Config config;
  config.exp_save_file_name = save_file_name;

  int result = 0;

  std::vector<std::unique_ptr<Experience>> experiences;
  for (const auto& state : builder.GetStates()) {
    auto legal_moves = state->GetLegalMoves();

    std::vector<std::pair<Move, float>> moves;
    for (auto m : legal_moves) {
      moves.push_back(std::make_pair(m, 1.0 / legal_moves.size()));
    }

    std::unique_ptr<Experience> experience = CreateExperience(
        std::make_unique<GameState>(*state), moves, (++result) % 2);
    experiences.push_back(std::move(experience));
  }

  {
    ExperienceSaver saver(&config);
    saver.SaveExperiences(experiences);

    // To make sure that the stream is flushed.
  }

  std::vector<std::unique_ptr<ExperienceSerialized>> serialized =
      DeserializeExperiences(save_file_name);

  ASSERT_EQ(experiences.size(), serialized.size());

  for (size_t i = 0; i < serialized.size(); i++) {
    EXPECT_TRUE(
        GameStateToTensor(*experiences[i]->state)
            .allclose(GameStateSerializedToTensor(serialized[i]->game_state)));

    torch::Tensor serialized_policy_vec =
        torch::from_blob(serialized[i]->policy_vec.data(), {73 * 8 * 8});
    EXPECT_TRUE(experiences[i]->policy.allclose(serialized_policy_vec));
    EXPECT_EQ(experiences[i]->result, serialized[i]->result);
  }
}
}  // namespace
}  // namespace chess
