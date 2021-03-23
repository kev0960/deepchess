#include "chess.h"
#include "dirichlet.h"
#include "train.h"

int main() {
  if (torch::cuda::cudnn_is_available()) {
    std::cout << "torch avail! " << std::endl;
  }

  chess::Config config("../config.json");
  config.PrintConfig();

  chess::Train trainer(&config);
  trainer.DoTrain();

  /*
  chess::Chess game;
  chess::ChessNN chess_nn(10, &device_manager);
  chess_nn->to(device);

  chess::DirichletDistribution dist(0.3);

  chess::Agent agent(&chess_nn, &dist);
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
