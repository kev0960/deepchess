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

json MoveVecToString(const std::vector<Move>& moves) {
  std::vector<std::string> str_moves;
  str_moves.reserve(moves.size());

  for (const auto& m : moves) {
    str_moves.push_back(m.Str());
  }

  return str_moves;
}

std::string GameResultToString(GameResult result) {
  switch (result) {
    case BLACK_WIN:
      return "BLACK_WIN";
    case WHITE_WIN:
      return "WHITE_WIN";
    default:
      return "DRAW";
  }
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
  server_runner_ = std::make_unique<std::thread>(&Server::ServerRunner, this);
}

void Server::ServerRunner() {
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
  evaluator_ = std::make_unique<Evaluator>(chess_nn_, config_,
                                           /*worker_manager=*/nullptr);
  evaluator_->StartInferenceWorker();

  agent_ = std::make_unique<Agent>(&dist_, config_, evaluator_.get(),
                                   /*worker_manager=*/nullptr, 0);

  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);

  std::cout << "Running server at port : " << config_->server_port << std::endl;
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
  json request = json::parse(request_str);

  RETURN_ERROR_IF_MISSING(request, action);
  if (action == "WorkerInfo") {
    return HandleWorkerInfo();
  } else if (action == "GameInfo") {
    std::optional<std::string> maybe_game_id = GetFromJson(request, "game_id");

    return HandleGameInfo(maybe_game_id);
  }

  RETURN_ERROR_IF_MISSING(request, game_id);

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

absl::StatusOr<std::string> Server::HandleWorkerInfo() {
  json result;

  std::vector<std::map<std::string, int>> worker_infos;
  for (int worker_id = 0; worker_id < config_->num_threads; worker_id++) {
    const auto& info =
        server_context_->GetWorkerManager()->GetWorkerInfo(worker_id);

    auto& worker_info = worker_infos.emplace_back();
    worker_info["current_game_total_move"] = info.current_game_total_move;
    worker_info["total_game_played"] = info.total_game_played;
  }

  result["worker_info"] = worker_infos;

  std::vector<std::map<std::string, int>> inference_worker_infos;
  for (int worker_id = 0; worker_id < config_->evaluator_worker_count;
       worker_id++) {
    const auto& info =
        server_context_->GetWorkerManager()->GetInferenceWorkerInfo(worker_id);

    auto& worker_info = inference_worker_infos.emplace_back();
    worker_info["total_inference_batch_size"] = info.total_inference_batch_size;
    worker_info["total_num_inference"] = info.total_num_inference;
  }

  result["inference_worker_info"] = inference_worker_infos;

  return result.dump();
}

absl::StatusOr<std::string> Server::HandleGameInfo(
    std::optional<std::string> game_id) {
  auto recorded_games = server_context_->GetGames();

  if (game_id) {
    size_t id = std::stoi(game_id.value());
    if (id >= recorded_games.size()) {
      return absl::InvalidArgumentError("Game Id is not correctly formed");
    }

    auto& game = recorded_games[id];

    json result;
    std::map<std::string, json> m;
    m["game_result"] = GameResultToString(game.first);
    m["moves"] = MoveVecToString(game.second);

    result["game"] = m;
    return result.dump();
  } else {
    // If the game id is not specified, then just send the list of played games.
    json result;

    std::vector<std::map<std::string, json>> games;
    for (const auto& [game_result, moves] : recorded_games) {
      auto& m = games.emplace_back();
      m["game_result"] = GameResultToString(game_result);
      m["moves"] = MoveVecToString(moves);
    }

    result["games"] = games;
    return result.dump();
  }
}

}  // namespace chess
