#include "agent.h"

#include <thread>

#include "mcts.h"

namespace chess {
namespace {

std::pair<Experience, Move> GetMoveForSelfPlay(
    std::unique_ptr<GameState> current, Evaluator* evaluator,
    DirichletDistribution* dirichlet, Config* config) {
  MCTS mcts(current.get(), evaluator, dirichlet, config);
  mcts.RunMCTS();

  Move best_move = mcts.MoveToMake(/*choose_best_move=*/false);
  torch::Tensor policy = mcts.GetPolicyVector();

  return std::make_pair(Experience{std::move(current), policy, 0}, best_move);
}

}  // namespace

Agent::Agent(ChessNN nn, DirichletDistribution* dirichlet, Config* config)
    : nn_(nn), dirichlet_(dirichlet), config_(config) {}

void Agent::Run() { DoSelfPlay(); }

void Agent::DoSelfPlay() {
  // Generate experiences.
  auto current = std::make_unique<GameState>(GameState::CreateInitGameState());

  Evaluator evaluator(nn_, config_);

  auto start = std::chrono::high_resolution_clock::now();

  int num_move = 0;
  while (num_move < config_->max_game_moves_until_draw) {
    // Checkmate! Game is over :)
    if (current->GetLegalMoves().empty()) {
      break;
    }

    if (current->IsDraw()) {
      break;
    }

    auto [experience, move] =
        GetMoveForSelfPlay(std::move(current), &evaluator, dirichlet_, config_);
    experiences_.push_back(std::make_unique<Experience>(std::move(experience)));

    current =
        std::make_unique<GameState>(experiences_.back()->state.get(), move);
    num_move++;

    fmt::print("{} (Moves: {}) ------ \n",
               std::hash<std::thread::id>{}(std::this_thread::get_id()),
               num_move);
    current->GetBoard().PrettyPrintBoard();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  fmt::print("Took {} ms", ms.count() / 1000.0);

  // When the game ends, "current" owns the game ending state, which is not
  // added to the experiences queue.
  if (num_move == config_->max_game_moves_until_draw || current->IsDraw()) {
    // This is draw.
    fmt::print("Game Is Over! : DRAW \n");
    current->GetBoard().PrettyPrintBoard();
    return;
  }

  GameState* last_state = current.get();
  PieceSide loser = last_state->WhoIsMoving();

  fmt::print("Game Is Over! : WHITE WIN? {} \n", loser == BLACK);
  last_state->GetBoard().PrettyPrintBoard();

  for (const auto& experience : experiences_) {
    if (experience->state->WhoIsMoving() == loser) {
      experience->result = -1;
    } else {
      experience->result = 1;
    }
  }
}

Move Agent::GetBestMove(const GameState& game_state) const {
  Evaluator evaluator(nn_, config_);

  MCTS mcts(&game_state, &evaluator, dirichlet_, config_);
  mcts.RunMCTS();

  return mcts.MoveToMake(/*choose_best_move=*/true);
}

}  // namespace chess
