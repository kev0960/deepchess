#ifndef SERVER_H
#define SERVER_H

#include <absl/status/statusor.h>

#include <memory>
#include <unordered_map>
#include <zmq.hpp>

#include "chess.h"
#include "config.h"
#include "game_state.h"
#include "server_context.h"

namespace chess {

// Server where the user can play.
class Server {
 public:
  Server(Config* config, ServerContext* server_context)
      : config_(config),
        chess_nn_(config->num_layer, config->num_filter),
        server_context_(server_context) {
    chess_nn_->to(config_->device);
  }

  absl::StatusOr<std::string> HandleRequest(std::string_view request);

  // Create a new game where the client takes client_side. If the client plays
  // black, then the server returns the first move of white.
  absl::StatusOr<Move> CreateNewGame(const std::string& game_id,
                                     PieceSide client_side);

  // Client makes the move.
  absl::StatusOr<std::string> DoMove(const std::string& game_id, Move move);

  // Handle WorkerInfo request.
  absl::StatusOr<std::string> HandleWorkerInfo();

  // Handle GameInfo request.
  absl::StatusOr<std::string> HandleGameInfo(
      std::optional<std::string> game_id);

  // Start the server.
  void RunServer();

 private:
  void ServerRunner();

  // Mapping between user_id to the current matches.
  std::unordered_map<std::string, std::vector<std::unique_ptr<GameState>>>
      matches_;

  Config* config_;

  UniformDistribution dist_;

  ChessNN chess_nn_;
  std::unique_ptr<Evaluator> evaluator_;
  std::unique_ptr<Agent> agent_;

  ServerContext* server_context_;
  std::unique_ptr<std::thread> server_runner_;
};

}  // namespace chess

#endif
