#include "serialize.h"

#include <fmt/core.h>

#include <filesystem>

namespace chess {
namespace {

namespace fs = std::filesystem;

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
    int policy_vec_size = exp->policy.sizes()[1];
    assert(policy_vec_size == 4672);

    std::lock_guard<std::mutex> lk(m_out_file_);
    DumpAsBinary(serialized_game_state, out_file_);
    DumpAsBinary(policy_vec, policy_vec_size * sizeof(float), out_file_);
    DumpAsBinary(exp->result, out_file_);
  }
}

void ExperienceSaver::ClearSavedExperiences() {
  out_file_.close();

  // Open the file again with write mode to clear contents.
  out_file_.open(config_->exp_save_file_name.c_str(), std::ios::binary);
}

std::vector<std::unique_ptr<ExperienceSerialized>> DeserializeExperiences(
    const std::string& file_name) {
  std::ifstream in(file_name.c_str(), std::ios::binary);
  if (!in.is_open()) {
    std::cout << "File " << file_name << " does not exist! \n";
    return {};
  }

  const size_t serialized_size = sizeof(ExperienceSerialized::game_state) +
                                 sizeof(ExperienceSerialized::policy_vec) +
                                 sizeof(ExperienceSerialized::result);

  size_t binary_size = fs::file_size(fs::path(file_name));
  if (binary_size % serialized_size != 0) {
    fmt::print(
        "Fatal error: {} is corrupted! File size is {}, not multiple of {}",
        file_name, binary_size, serialized_size);
    return {};
  }

  std::vector<std::unique_ptr<ExperienceSerialized>> exps;
  while (exps.size() < binary_size / serialized_size) {
    exps.push_back(std::make_unique<ExperienceSerialized>());
    auto& exp = exps.back();

    ReadFromBinary(exp->game_state, in);

    if (!in) {
      fmt::print(
          "Fatal error: {} is corrupted! Failed while reading "
          "GameStateSerialized at {}",
          file_name, exps.size());
      return {};
    }

    ReadFromBinary(exp->policy_vec, in);

    if (!in) {
      fmt::print(
          "Fatal error: {} is corrupted! Failed while reading "
          "Policy vec at {}",
          file_name, exps.size());
      return {};
    }

    ReadFromBinary(exp->result, in);
  }

  return exps;
}

}  // namespace chess
