#include "server.h"

#include <nlohmann/json.hpp>

#include "util.h"

namespace chess {
namespace {

using json = nlohmann::json;

std::optional<std::string> GetFromJson(const json& j,
                                       const std::string& field) {
  if (!j.count(field)) {
    return std::nullopt;
  }

  return j[field].get<std::string>();
}

std::string MoveToJsonString(Move m) {
  return absl::StrCat("{'move' : '", m.Str(), "'}");
}

// Defines the variable name config.
#define RETURN_ERROR_IF_MISSING(request, config)                             \
  std::optional<std::string> maybe_##config = GetFromJson(request, #config); \
  if (!maybe_##config) {                                                     \
    return absl::InvalidArgumentError("Error Missing config " #config);      \
  }                                                                          \
  std::string config = maybe_##config.value();
}  // namespace

void Server::RunServer() {
  // Set up the model.
  std::string model_name = config_->existing_model_name;
  if (IsFileExist(model_name)) {
    torch::load(chess_nn_, model_name);
  } else {
    std::cout << model_name << " is not found" << std::endl;
    model_name = "CurrentBest.pt";
    torch::save(chess_nn_, model_name);
  }

  // Set up the evaluator.
  evaluator_ = std::make_unique<Evaluator>(chess_nn_, config_);
  evaluator_->StartInferenceWorker();

  agent_ = std::make_unique<Agent>(&dist_, config_, evaluator_.get(), 0);

  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);

  socket.bind("tcp://*:" + config_->server_port);

  while (true) {
    zmq::message_t req;
    auto result = socket.recv(req);

    if (!result.has_value()) {
      std::cerr << "Received the invalid request." << std::endl;
      socket.send(zmq::buffer(R"({"result" : "Error"})"));
      continue;
    }

    absl::StatusOr<std::string> response = HandleRequest(req.to_string_view());
    if (response.ok()) {
      socket.send(zmq::buffer(response.value()));
    } else {
      socket.send(zmq::buffer(
          absl::StrCat("{'result' : '", response.status().ToString(), "'}")));
    }
  }
}

absl::StatusOr<std::string> Server::HandleRequest(
    std::string_view request_str) {
  json request(request_str);

  RETURN_ERROR_IF_MISSING(request, game_id);
  RETURN_ERROR_IF_MISSING(request, action);

  if (action == "Create") {
    RETURN_ERROR_IF_MISSING(request, client_side);
    PieceSide side = client_side == "white" ? WHITE : BLACK;

    absl::StatusOr<Move> move = CreateNewGame(game_id, side);
    if (!move.ok()) {
      return move.status();
    }

    return MoveToJsonString(move.value());
  } else if (action == "Move") {
    RETURN_ERROR_IF_MISSING(request, move);
    return DoMove(game_id, Move::MoveFromString(move));
  } else if (action == "Resign") {
    if (matches_.find(game_id) == matches_.end()) {
      return absl::InvalidArgumentError("Game does not exist");
    }

    matches_.erase(matches_.find(game_id));
    return "{'result' : 'lost'}";
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unknown action [", action, "]"));
}

absl::StatusOr<Move> Server::CreateNewGame(const std::string& game_id,
                                           PieceSide client_side) {
  if (matches_.find(game_id) != matches_.end()) {
    return absl::InvalidArgumentError("Game already exists");
  }

  auto& states = matches_[game_id];
  states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  if (client_side == WHITE) {
    // Return dummy move.
    return Move(0, 0, 0, 0);
  } else {
    Move move = agent_->GetBestMove(*states.back());
    states.push_back(std::make_unique<GameState>(states.back().get(), move));

    return move;
  }
}

absl::StatusOr<std::string> Server::DoMove(const std::string& game_id,
                                           Move move) {
  if (matches_.find(game_id) == matches_.end()) {
    return absl::InvalidArgumentError("Game does not exist");
  }

  auto& states = matches_[game_id];

  const GameState& current = *states.back();
  auto legal_moves = current.GetLegalMoves();

  if (std::find(legal_moves.begin(), legal_moves.end(), move) ==
      legal_moves.end()) {
    return absl::InvalidArgumentError("Not legal move.");
  }

  states.push_back(std::make_unique<GameState>(&current, move));

  if (states.back()->IsDraw()) {
    // Game is done; Remove from the matches.
    matches_.erase(matches_.find(game_id));
    return "{'result' : 'draw'}";
  } else if (states.back()->GetLegalMoves().empty()) {
    matches_.erase(matches_.find(game_id));
    return "{'result' : 'win'}";
  }

  Move computer_move = agent_->GetBestMove(*states.back());
  states.push_back(
      std::make_unique<GameState>(states.back().get(), computer_move));

  return MoveToJsonString(computer_move);
}

}  // namespace chess
