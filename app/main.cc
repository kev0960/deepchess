#include "chess.h"
#include "server.h"
#include "train.h"

int main() {
  if (torch::cuda::cudnn_is_available()) {
    std::cout << "torch avail! " << std::endl;
  }

  chess::Config config("../config.json");
  config.PrintConfig();

  chess::Server server(&config);

  chess::Train trainer(&config);
  trainer.DoTrain();

  
  /*
  chess::Chess game(&config);
  chess::ChessNN chess_nn(15);
  chess_nn->to(config.device);

  chess::UniformDistribution dist;
  chess::Evaluator eval(chess_nn, &config);
  eval.StartInferenceWorker();

  chess::Agent agent(&dist, &config, &eval, 0);
  auto result = game.PlayChessWithHuman(&agent, chess::WHITE);
  switch (result) {
    case chess::DRAW:
      std::cout << "Draw! \n";
      break;
    case chess::WHITE_WIN:
      std::cout << "White won! \n";
      break;
    case chess::BLACK_WIN:
      std::cout << "Black won! \n";
      break;
  }
  */  
}
