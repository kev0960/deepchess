#include "agent.h"

#include <thread>

#include "mcts.h"

namespace chess {
namespace {

std::pair<Experience, Move> GetMoveForSelfPlay(
    std::unique_ptr<GameState> current, Evaluator* evaluator,
    DirichletDistribution* dirichlet, Config* config, int worker_id) {
  MCTS mcts(current.get(), evaluator, dirichlet, config, worker_id);
  mcts.RunMCTS();

  Move best_move = mcts.MoveToMake(/*choose_best_move=*/false);
  torch::Tensor policy = mcts.GetPolicyVector();

  return std::make_pair(Experience{std::move(current), policy, 0}, best_move);
}

}  // namespace

Agent::Agent(DirichletDistribution* dirichlet, Config* config,
             Evaluator* evaluator, int worker_id)
    : dirichlet_(dirichlet),
      config_(config),
      evaluator_(evaluator),
      worker_id_(worker_id) {}

void Agent::Run() { DoSelfPlay(); }

void Agent::DoSelfPlay() {
  // Generate experiences.
  auto current = std::make_unique<GameState>(GameState::CreateInitGameState());
  auto start = std::chrono::high_resolution_clock::now();

  int num_move = 0;
  while (num_move < config_->max_game_moves_until_draw) {
    // Game is over :)
    if (current->GetLegalMoves().empty()) {
      break;
    }

    if (current->IsDraw()) {
      break;
    }

    auto [experience, move] = GetMoveForSelfPlay(
        std::move(current), evaluator_, dirichlet_, config_, worker_id_);
    experiences_.push_back(std::make_unique<Experience>(std::move(experience)));

    current =
        std::make_unique<GameState>(experiences_.back()->state.get(), move);
    num_move++;

    if (config_->show_self_play_boards) {
      fmt::print("Worker [{}] (Moves: {}) ------ \n", worker_id_, num_move);
      current->GetBoard().PrettyPrintBoard();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  fmt::print("Took {:.3f} minutes ... {} secs per move (Total : {} moves) \n",
             ms.count() / 1000.0 / 60, ms.count() / num_move / 1000.0,
             num_move);

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
  MCTS mcts(&game_state, evaluator_, dirichlet_, config_, worker_id_);
  mcts.RunMCTS();

  return mcts.MoveToMake(/*choose_best_move=*/true);
}

}  // namespace chess
