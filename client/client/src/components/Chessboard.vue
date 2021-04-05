<template>
  <div ref="board" id="chessboard"></div>
</template>

<script>
import { Chessboard } from "cm-chessboard/src/cm-chessboard/Chessboard.js";

export default {
  mounted() {
    this.chessboard = new Chessboard(this.$refs.board, {position: "start"});
    this.chessboard.enableMoveInput();
  }
}
</script>

<style>
@import "cm-chessboard/styles/cm-chessboard.css";

#chessboard {
  width: 50%;
}
</style>