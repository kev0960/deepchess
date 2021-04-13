#include "server_context.h"

namespace chess {
namespace {

GameResult ResultToGameResult(float result) {
  if (result == -1) {
    return BLACK_WIN;
  } else if (result == 1) {
    return WHITE_WIN;
  }

  return DRAW;
}

}  // namespace

void ServerContext::RecordGame(
    const std::vector<std::unique_ptr<Experience>>& experiences) {
  std::vector<Move> moves;
  for (const auto& exp : experiences) {
    moves.push_back(exp->state->LastMove());
  }

  std::lock_guard lk(m_game_);
  games_.push_back(
      std::make_pair(ResultToGameResult(experiences[0]->result), moves));
}

std::vector<std::pair<GameResult, std::vector<Move>>>
ServerContext::GetGames() {
  std::lock_guard lk(m_game_);
  return games_;
}

void ServerContext::DeleteRecordedGames() {
  std::lock_guard lk(m_game_);
  games_.clear();
}

}  // namespace chess
