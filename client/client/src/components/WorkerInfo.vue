<template>
  <div class="worker-info-box">
    <div class="worker-id">Worker [{{ worker_id }}]</div>
    <div>
      {{ info.current_game_total_move }} and
      {{ info.total_game_played }}
    </div>
  </div>
</template>

<script>
import { computed } from "vue";

export default {
  props: ["worker_info", "worker_id"],
  setup(props) {
    const info = computed({ get: () => props.worker_info });
    return { info };
  },
};
</script>

<style>
.worker-info-box {
  display: flex;
  flex-direction: row;
  border: 1px solid black;
  padding: 10px;
  font-size: 13px;
  font-family: 'monospace';
}

.worker-id {
  margin-right: 10px;
  width: 90px;
  text-align: left;
}

</style>