#include "agent.h"

#include <thread>

#include "mcts.h"

namespace chess {
namespace {

constexpr int kMaxGameMoves = 300;
constexpr int kMCTSNumIter = 1000;

std::pair<Experience, Move> GetMove(GameState* current, Evaluator* evaluator,
                                    DirichletDistribution* dirichlet) {
  MCTS mcts(current, evaluator, dirichlet);
  mcts.RunMCTS(kMCTSNumIter);

  Move best_move = mcts.MoveToMake();
  torch::Tensor policy = mcts.GetPolicyVector();

  return std::make_pair(Experience{current, policy, 0}, best_move);
}

}  // namespace

Agent::Agent(ChessNN nn, DirichletDistribution* dirichlet,
             DeviceManager* device_manager)
    : nn_(nn), dirichlet_(dirichlet), device_manager_(device_manager) {}

void Agent::Run() { DoSelfPlay(); }

void Agent::DoSelfPlay() {
  // Generate experiences.
  states_.push_back(
      std::make_unique<GameState>(GameState::CreateInitGameState()));

  Evaluator evaluator(nn_, device_manager_);

  int num_move = 0;
  while (num_move < kMaxGameMoves) {
    GameState* current = states_.back().get();

    // Checkmate! Game is over :)
    if (current->GetLegalMoves().empty()) {
      break;
    }

    if (current->IsDraw()) {
      break;
    }

    auto [experience, move] = GetMove(current, &evaluator, dirichlet_);
    experiences_.push_back(std::make_unique<Experience>(experience));

    states_.push_back(std::make_unique<GameState>(current, move));
    num_move++;

    std::cout << std::this_thread::get_id() << " -----\n";
    states_.back()->GetBoard().PrettyPrintBoard();
  }

  GameState* last_state = states_.back().get();
  if (num_move == kMaxGameMoves || last_state->IsDraw()) {
    // This is draw.
    fmt::print("Game Is Over! : DRAW \n");
    states_.back()->GetBoard().PrettyPrintBoard();
    return;
  }

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

Move Agent::GetBestMove(const GameState& game_state, int num_mcts_iter) const {
  Evaluator evaluator(nn_, device_manager_);

  MCTS mcts(&game_state, &evaluator, dirichlet_);
  mcts.RunMCTS(num_mcts_iter);

  return mcts.MoveToMake();
}

}  // namespace chess
