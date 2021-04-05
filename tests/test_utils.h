#ifndef TESTS_TEST_UTILS_H
#define TESTS_TEST_UTILS_H

#include <memory>
#include <vector>

#include "absl/strings/str_split.h"
#include "agent.h"
#include "board.h"
#include "game_state.h"

namespace chess {

class GameStateBuilder {
 public:
  GameStateBuilder(GameState state = GameState::CreateInitGameState()) {
    states_.push_back(std::make_unique<GameState>(state));
  }

  GameStateBuilder& DoMove(Move move);

  const std::vector<std::unique_ptr<GameState>>& GetStates() const {
    return states_;
  }

  std::unique_ptr<GameState> ReleaseStateAt(int i) {
    return std::move(states_[i]);
  }

 private:
  std::vector<std::unique_ptr<GameState>> states_;
};

// Notation looks like this:
// rnbqkbnr
// pppppppp
//
//
//
//
// PPPPPPPP
// RNBQKBNR
Board BoardFromNotation(std::string_view notation);

std::unique_ptr<Experience> CreateExperience(
    std::unique_ptr<GameState> state, std::vector<std::pair<Move, float>> move,
    float reward);

}  // namespace chess

#endif
