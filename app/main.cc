#include "chess.h"
#include "server.h"
#include "server_context.h"
#include "train.h"

int main() {
  if (torch::cuda::cudnn_is_available()) {
    std::cout << "torch avail! " << std::endl;
  }

  chess::Config config("../config.json");
  config.PrintConfig();

  chess::ServerContext server_context(&config);
  chess::Server server(&config, &server_context);

  if (config.run_server) {
    server.RunServer();
  }

  if (config.do_train) {
    chess::Train trainer(&config, &server_context);
    trainer.DoTrain();
  } else {
    chess::Chess game(&config);
    chess::ChessNN chess_nn(config.num_layer, config.num_filter);
    chess_nn->to(config.device);

    chess::UniformDistribution dist;
    chess::Evaluator eval(chess_nn, &config, server_context.GetWorkerManager());
    eval.StartInferenceWorker();

    chess::Agent agent(&dist, &config, &eval, server_context.GetWorkerManager(),
                       0);
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
  }
}
