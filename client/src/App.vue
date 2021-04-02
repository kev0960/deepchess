<template>
  <div>
    <img alt="Vue logo" src="./assets/logo.png" />
    <p>{{ greeting }} World! {{ count }}</p>
    <a v-bind:href="url">hi</a>
    <button @click="increment">Button</button>
    <button @click="createBoard">Create</button>

    <div ref="board" id='chessboard'></div>
  </div>
</template>

<script>
import { Chessboard, INPUT_EVENT_TYPE } from "cm-chessboard/src/cm-chessboard/Chessboard.js";

function inputHandler(event) {
    switch (event.type) {
        case INPUT_EVENT_TYPE.moveStart:
            return true
        case INPUT_EVENT_TYPE.moveDone:
            return true
        case INPUT_EVENT_TYPE.moveCanceled:
            return false;
    }
}

export default {
  data() {
    return {
      greeting: "Hello",
      url: "http://hi",
      count: 0,
    };
  },
  methods: {
    increment() {
      this.count++;
    },
    createBoard() {
      this.board = new Chessboard(this.$refs.board, {
        position: "rn2k1r1/ppp1pp1p/3p2p1/5bn1/P7/2N2B2/1PPPPP2/2BNK1RR",
      });
      this.board.enableMoveInput(inputHandler);
    },
  },
};
</script>

<style>
@import "cm-chessboard/styles/cm-chessboard.css";

#chessboard {
  width: 50%;
}
</style>