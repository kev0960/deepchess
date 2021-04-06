#include "serialize.h"

namespace chess {
namespace {

template <typename T>
void DumpAsBinary(const T* data, int size, std::ofstream& out) {
  out.write(reinterpret_cast<const char*>(data), size);
}

template <typename T>
void DumpAsBinary(const T& data, std::ofstream& out) {
  return DumpAsBinary(&data, sizeof(T), out);
}

}  // namespace

ExperienceSaver::ExperienceSaver(Config* config)
    : config_(config), out_file_(config->exp_save_file_name.c_str()) {}

void ExperienceSaver::SaveExperiences(
    const std::vector<std::unique_ptr<Experience>>& exps) {
  for (const auto& exp : exps) {
    GameStateSerialized serialized_game_state =
        exp->state->GetGameStateSerialized();

    float* policy_vec = exp->policy.data_ptr<float>();
    int policy_vec_size = exp->policy.sizes()[0];

    std::lock_guard<std::mutex> lk(m_out_file_);
    DumpAsBinary(serialized_game_state, out_file_);
    DumpAsBinary(policy_vec, policy_vec_size, out_file_);
    DumpAsBinary(exp->result, out_file_);
  }
}

}  // namespace chess
