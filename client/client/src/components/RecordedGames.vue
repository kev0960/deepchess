<template>
  <div class='recorded-game-container'>
    <div v-for="(game, index) in recorded_games" :key="index"></div>
  </div>
</template>

<script>
import { defineComponent } from "@vue/composition-api";

export default defineComponent({
  data() {
    return {
      recorded_games: [],
    };
  },
  mounted() {
    fetch("http://localhost:3001/game-info", {
      method: "POST",
      body: JSON.stringify({}),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        this.recorded_games = data;
      });
  },
});
</script>
