#include "server.h"

#include <nlohmann/json.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_utils.h"

namespace chess {
namespace {

using json = nlohmann::json;

TEST(ServerTest, HandleGameInfoRequest) {
  Config config;
  ServerContext server_context(&config);

  std::vector<std::unique_ptr<Experience>> experiences;
  GameStateBuilder builder;
  builder
      .DoMove(Move(7, 1, 5, 2))  // Nb3
      .DoMove(Move(0, 1, 2, 2))  // Nc6
      /* White Knight goes back to original position. */
      .DoMove(Move(5, 2, 7, 1))
      /* Black Knight goes back to original position. */
      .DoMove(Move(2, 2, 0, 1));

  experiences.push_back(
      CreateExperience(builder.ReleaseStateAt(0),
                       {{Move(7, 1, 5, 2), 0.9}, {Move(6, 1, 4, 1), 0.1}}, 1));
  experiences.push_back(
      CreateExperience(builder.ReleaseStateAt(1),
                       {{Move(0, 1, 2, 2), 0.6}, {Move(1, 3, 3, 3), 0.4}}, -1));
  experiences.push_back(
      CreateExperience(builder.ReleaseStateAt(2),
                       {{Move(5, 2, 7, 1), 0.8}, {Move(6, 3, 4, 3), 0.1}}, 1));
  experiences.push_back(CreateExperience(builder.ReleaseStateAt(3),
                                         {{Move(2, 2, 0, 1), 0.55},
                                          {Move(1, 5, 3, 5), 0.4},
                                          {Move(1, 7, 3, 7), 0.05}},
                                         -1));
  server_context.RecordGame(experiences);

  Server server(&config, &server_context);

  json request;
  request["action"] = "GameInfo";

  absl::StatusOr<std::string> result = server.HandleRequest(request.dump());

  json expected = R"(
    {
      "games" : [
        {
          "game_result" : "WHITE_WIN",
          "moves" : ["a8a8", "b1c3", "b8c6", "c3b1"]
        }
      ]
    }
  )"_json;

  EXPECT_TRUE(result.ok());
  EXPECT_EQ(json::parse(result.value()), expected);

  // With the game_id in the request.
  request["game_id"] = "0";
  result = server.HandleRequest(request.dump());

  json expected2 = R"(
    {
      "game" : 
        {
          "game_result" : "WHITE_WIN",
          "moves" : ["a8a8", "b1c3", "b8c6", "c3b1"]
        }
    }
  )"_json;

  EXPECT_TRUE(result.ok());
  EXPECT_EQ(json::parse(result.value()), expected2);
}

TEST(ServerTest, HandleWorkerInfoRequest) {
  Config config;
  config.num_threads = 4;
  config.evaluator_worker_count = 2;

  ServerContext server_context(&config);

  WorkerManager* worker_manager = server_context.GetWorkerManager();
  worker_manager->GetInferenceWorkerInfo(0).total_inference_batch_size = 4;
  worker_manager->GetInferenceWorkerInfo(0).total_num_inference = 24;

  worker_manager->GetWorkerInfo(1).current_game_total_move = 32;
  worker_manager->GetWorkerInfo(1).total_game_played = 50;

  Server server(&config, &server_context);

  json request = R"({"action" : "WorkerInfo"})"_json;
  json expected = R"(
   {
   "inference_worker_info": [
        {
            "total_inference_batch_size": 4,
            "total_num_inference": 24
        },
        {
            "total_inference_batch_size": 0,
            "total_num_inference": 0
        }
    ],
    "worker_info": [
        {
            "current_game_total_move": 0,
            "total_game_played": 0
        },
        {
            "current_game_total_move": 32,
            "total_game_played": 50
        },
        {
            "current_game_total_move": 0,
            "total_game_played": 0
        },
        {
            "current_game_total_move": 0,
            "total_game_played": 0
        }
    ]
   } 
  )"_json;

  absl::StatusOr<std::string> result = server.HandleRequest(request.dump());

  EXPECT_TRUE(result.ok());
  EXPECT_EQ(json::parse(result.value()), expected);
}

}  // namespace
}  // namespace chess

