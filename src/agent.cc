#include "agent.h"

#include "mcts.h"

namespace chess {
namespace {

constexpr int kTotalNumIter = 1000;

/*
std::unique_ptr<GameState> GetNextState(GameState* current,
                                        Evaluator* evaluator) {
  MCTS mcts(current, evaluator);
  for (int i = 0; i < kTotalNumIter; i ++) {
    mcts.RunMCTS();
  }

}
*/

}  // namespace

Agent::Agent() : nn_(50, 117) {}

void Agent::Run() {
  for (int num_iter = 0; num_iter < kTotalNumIter; num_iter++) {
    DoSelfPlay();
  }
}

void Agent::DoSelfPlay() {
  std::vector<std::unique_ptr<GameState>> current_game_states;

  // Generate experiences.
  current_game_states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  //  GameState* current_game = current_game_states.back().get();
  //  PieceSide current_turn = PieceSide::WHITE;
}

}  // namespace chess
