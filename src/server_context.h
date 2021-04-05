#ifndef SERVER_CONTEXT_H
#define SERVER_CONTEXT_H

#include <thread>

#include "chess.h"
#include "config.h"
#include "move.h"
#include "worker_manager.h"

namespace chess {

// The class that contains every information that the server needs.
// This will be shared across training framework.
class ServerContext {
 public:
  ServerContext(Config* config) : worker_manager_(config) {}

  WorkerManager* GetWorkerManager() { return &worker_manager_; }

  void RecordGame(const std::vector<std::unique_ptr<Experience>>& experiences);
  std::vector<std::pair<GameResult, std::vector<Move>>> GetGames();

 private:
  WorkerManager worker_manager_;

  std::mutex m_game_;
  std::vector<std::pair<GameResult, std::vector<Move>>> games_;
};

}  // namespace chess

#endif
