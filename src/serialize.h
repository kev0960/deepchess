#ifndef SERIALIZE_H
#define SERIALIZE_H

#include "agent.h"
#include "config.h"
#include <fstream>

namespace chess {

class ExperienceSaver {
 public:
  ExperienceSaver(Config* config);

  void SaveExperiences(const std::vector<std::unique_ptr<Experience>>& exps);

 private:
  Config* config_;

  std::mutex m_out_file_;
  std::ofstream out_file_;
};

}  // namespace chess

#endif
