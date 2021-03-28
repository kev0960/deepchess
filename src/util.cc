#include "util.h"

#include <execinfo.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>

namespace chess {

void PrintStackTrace() {
  void* arr[100];
  size_t depth = backtrace(arr, 100);

  backtrace_symbols_fd(arr, depth, STDERR_FILENO);
}

bool IsFileExist(const std::string& file_name) {
  std::ifstream in(file_name.c_str());
  return in.is_open();
}

}  // namespace chess
