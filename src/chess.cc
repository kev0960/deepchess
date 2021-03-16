#include "chess.h"

#include <fmt/core.h>

#include <iostream>

#include "game_state.h"
#include "move.h"

namespace chess {
namespace {

constexpr int kAgentEvalMCTSIter = 600;
constexpr int kMaxGameMoves = 1000;

Move StringToMove(std::string_view str_move) {
  if (str_move.size() < 4) {
    return Move(0, 0, 0, 0);
  }

  int from_col = str_move[0] - 'a';
  int from_row = 7 - (str_move[1] - '1');

  int to_col = str_move[2] - 'a';
  int to_row = 7 - (str_move[3] - '1');

  if (str_move.size() == 5) {
    Promotion promotion = NO_PROMOTE;
    switch (str_move[4]) {
      case 'q':
        promotion = PROMOTE_QUEEN;
        break;
      case 'n':
        promotion = PROMOTE_KNIGHT;
        break;
      case 'b':
        promotion = PROMOTE_BISHOP;
        break;
      case 'r':
        promotion = PROMOTE_ROOK;
        break;
    }

    return Move(from_row, from_col, to_row, to_col, promotion);
  }

  return Move(from_row, from_col, to_row, to_col);
}

Move GetMoveFromUser(std::vector<Move> legal_moves) {
  while (true) {
    std::string input;
    std::getline(std::cin, input);

    Move m = StringToMove(input);
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
  while (current->TotalMoveCount() < kMaxGameMoves) {
    if (current->IsDraw()) {
      return DRAW;
    }

    if (current->GetLegalMoves().empty()) {
      return current->WhoIsMoving() == WHITE ? BLACK_WIN : WHITE_WIN;
    }

    Move best_move(0, 0, 0, 0);
    if (current->WhoIsMoving() == WHITE) {
      best_move = white->GetBestMove(*current, kAgentEvalMCTSIter);
    } else {
      best_move = black->GetBestMove(*current, kAgentEvalMCTSIter);
    }

    states.push_back(std::make_unique<GameState>(current, best_move));
    current = states.back().get();
  }

  return DRAW;
}

GameResult Chess::PlayChessWithHuman(const Agent* agent, PieceSide my_color) {
  std::vector<std::unique_ptr<GameState>> states;
  states.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  GameState* current = states.back().get();
  while (current->TotalMoveCount() < kMaxGameMoves) {
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
      best_move = agent->GetBestMove(*current, kAgentEvalMCTSIter);
    }

    states.push_back(std::make_unique<GameState>(current, best_move));
    current = states.back().get();
  }

  return DRAW;
}

}  // namespace chess
