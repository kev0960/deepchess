#include "serialize.h"

#include <fmt/core.h>

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

template <typename T>
void ReadFromBinary(T* data, int size, std::ifstream& in) {
  in.read(reinterpret_cast<char*>(data), size);
}

template <typename T>
void ReadFromBinary(T& data, std::ifstream& in) {
  ReadFromBinary(&data, sizeof(T), in);
}

}  // namespace

ExperienceSaver::ExperienceSaver(Config* config)
    : config_(config),
      out_file_(config->exp_save_file_name.c_str(), std::ios::binary) {}

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

std::vector<std::unique_ptr<ExperienceSerialized>> DeserializeExperiences(
    const std::string& file_name) {
  std::ifstream in(file_name.c_str());
  if (in.is_open()) {
    std::cout << "File " << file_name << " does not exist! \n";
    return {};
  }

  std::vector<std::unique_ptr<ExperienceSerialized>> exps;
  while (in) {
    auto& exp = exps.emplace_back();
    ReadFromBinary(exp->game_state, in);

    if (!in) {
      fmt::print(
          "Fatal error: {} is corrupted! Failed while reading "
          "GameStateSerialized at {}",
          exps.size());
      return {};
    }

    ReadFromBinary(exp->policy_vec, in);

    if (!in) {
      fmt::print(
          "Fatal error: {} is corrupted! Failed while reading "
          "Policy vec at {}",
          exps.size());
      return {};
    }

    ReadFromBinary(exp->result, in);
  }

  return exps;
}

}  // namespace chess
