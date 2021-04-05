const express = require('express');
const zmq = require("zeromq");

const port = 3001;
const trainer_port = 8888;
const sock = new zmq.Request;

const app = express();
app.use(express.urlencoded({ extended: false }));
app.use(express.json());
app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "http://localhost:3000");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

function InitZmq() {
  console.log("Accessing server port at", "tcp://*:" + trainer_port);
  sock.connect("tcp://127.0.0.1:" + trainer_port);
}

app.post('/worker-info', async (req, res) => {
  let server_request = {
    "action": "WorkerInfo"
  };
  await sock.send(JSON.stringify(server_request));

  const [result] = await sock.receive();
  console.log(result.toString());

  return res.send(result);
});

app.post('/game-info', async (req, res) => {
  let game_id = req.body.game_info;
  let server_request = {
    "action": "GameInfo"
  };

  if (game_id) {
    server_request.game_id = "" + game_id;
  }

  const [result] = await sock.receive();
  console.log(result);

  return res.send(result);
});

app.listen(port, () => {
  InitZmq();

  console.log(`Example app listening at http://localhost:${port}`);
});