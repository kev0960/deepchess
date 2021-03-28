#include "chess.h"

#include <fmt/core.h>

#include <iostream>

#include "game_state.h"
#include "move.h"

namespace chess {
namespace {

Move GetMoveFromUser(std::vector<Move> legal_moves) {
  while (true) {
    std::string input;
    std::getline(std::cin, input);

    Move m = Move::MoveFromString(input);
    if (std::find(legal_moves.begin(), legal_moves.end(), m) !=
        legal_moves.end()) {
      return m;
    }

    fmt::print("Your move {} is not valid move. Please select among", input);
    for (auto& move : legal_moves) {
      fmt::print("{} ", move.Str());
    }
  }
}

}  // namespace

GameResult Chess::PlayChessBetweenAgents(const Agent* white,
                                         const Agent* black) {
  std::vector<std::unique_ptr<GameState>> states;
  states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  GameState* current = states.back().get();
  while (current->TotalMoveCount() < config_->max_game_moves_until_draw) {
    if (current->IsDraw()) {
      return DRAW;
    }

    if (current->GetLegalMoves().empty()) {
      return current->WhoIsMoving() == WHITE ? BLACK_WIN : WHITE_WIN;
    }

    Move best_move(0, 0, 0, 0);
    if (current->WhoIsMoving() == WHITE) {
      best_move = white->GetBestMove(*current);
    } else {
      best_move = black->GetBestMove(*current);
    }

    states.push_back(std::make_unique<GameState>(current, best_move));
    current = states.back().get();

    if (config_->show_self_play_boards) {
      fmt::print("Agent Self Play Worker [{}] .. at {} moves \n",
                 white->WorkerId(), current->TotalMoveCount());
      current->GetBoard().PrettyPrintBoard();
    }
  }

  return DRAW;
}

GameResult Chess::PlayChessWithHuman(const Agent* agent, PieceSide my_color) {
  std::vector<std::unique_ptr<GameState>> states;
  states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  GameState* current = states.back().get();
  while (current->TotalMoveCount() < config_->max_game_moves_until_draw) {
    if (current->IsDraw()) {
      return DRAW;
    }

    if (current->GetLegalMoves().empty()) {
      return current->WhoIsMoving() == WHITE ? BLACK_WIN : WHITE_WIN;
    }

    Move best_move(0, 0, 0, 0);
    if (current->WhoIsMoving() == my_color) {
      current->GetBoard().PrettyPrintBoard();
      best_move = GetMoveFromUser(current->GetLegalMoves());
    } else {
      best_move = agent->GetBestMove(*current);
    }

    states.push_back(std::make_unique<GameState>(current, best_move));
    current = states.back().get();
  }

  return DRAW;
}

}  // namespace chess
