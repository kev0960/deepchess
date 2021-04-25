#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <fstream>

#include "agent.h"
#include "config.h"

namespace chess {

struct ExperienceSerialized {
  GameStateSerialized game_state;
  std::array<float, 73 * 8 * 8> policy_vec;
  float result;
};

// Save the generated experiences.
class ExperienceSaver {
 public:
  ExperienceSaver(Config* config);

  void ClearSavedExperiences();
  void SaveExperiences(const std::vector<std::unique_ptr<Experience>>& exps);

 private:
  Config* config_;

  std::mutex m_out_file_;
  std::ofstream out_file_;
};

std::vector<std::unique_ptr<ExperienceSerialized>> DeserializeExperiences(
    const std::string& file_name);

}  // namespace chess

#endif
