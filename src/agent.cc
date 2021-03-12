#include "agent.h"

#include "mcts.h"

namespace chess {
namespace {

constexpr int kTotalNumIter = 1000;
constexpr int kMaxGameMoves = 1000;

std::pair<Experience, Move> GetMove(GameState* current, Evaluator* evaluator) {
  MCTS mcts(current, evaluator);
  for (int i = 0; i < kTotalNumIter; i++) {
    mcts.RunMCTS();
  }

  Move best_move = mcts.MoveToMake();
  torch::Tensor policy = mcts.GetPolicyVector();

  return std::make_pair(Experience{current, policy, 0}, best_move);
}

}  // namespace

Agent::Agent() : nn_(50, 117) {}

void Agent::Run() {
  for (int num_iter = 0; num_iter < kTotalNumIter; num_iter++) {
    DoSelfPlay();
  }
}

void Agent::DoSelfPlay() {
  std::vector<std::unique_ptr<GameState>> current_game_states;
  std::vector<std::unique_ptr<Experience>> current_experiences;

  // Generate experiences.
  current_game_states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  Evaluator evaluator(&nn_);

  int num_move = 0;
  while (num_move < kMaxGameMoves) {
    GameState* current = current_game_states.back().get();

    // Checkmate! Game is over :)
    if (current->GetLegalMoves().empty()) {
      break;
    }

    auto [experience, move] = GetMove(current, &evaluator);
    current_experiences.push_back(std::make_unique<Experience>(experience));

    current_game_states.push_back(std::make_unique<GameState>(current, move));
    num_move++;
  }

  GameState* last_state = current_game_states.back().get();
  PieceSide loser = last_state->WhoIsMoving();

  for (const auto& experience : current_experiences) {
    if (experience->state->WhoIsMoving() == loser) {
      experience->result = -1;
    } else {
      experience->result = 1;
    }
  }
}

}  // namespace chess
