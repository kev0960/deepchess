<template>
  <div>
    <div class="worker-container">
      <div
        v-for="(worker_info, index) in worker_infos.worker_info"
        :key="index"
      >
        <WorkerInfo
          :worker_info="worker_info"
          :worker_id="index + 1"
        ></WorkerInfo>
      </div>
    </div>
  </div>
</template>

<script>
import Chessboard from "./components/Chessboard.vue";
import WorkerInfo from "./components/WorkerInfo.vue";

export default {
  components: {
    Chessboard,
    WorkerInfo,
  },
  data() {
    return {
      worker_infos: null,
      worker_info_poll: null,
    };
  },
  methods: {
    workerInfoPoll() {
      this.worker_info_poll = setInterval(() => {
        console.log("Sending..");
        fetch("http://localhost:3001/worker-info", {
          method: "POST",
          body: JSON.stringify({}),
        })
          .then((res) => res.json())
          .then((data) => {
            console.log(data);
            this.worker_infos = data;
          });
      }, 1000);
    },
  },
  created() {
    this.workerInfoPoll();
  },
};
</script>

<style>
.worker-container {
  display: grid;
  grid-template-columns: auto auto auto auto;
}
</style>