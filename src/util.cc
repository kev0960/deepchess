#include "util.h"

#include <execinfo.h>
#include <unistd.h>

#include <cstdio>

namespace chess {

void PrintStackTrace() {
  void* arr[100];
  size_t depth = backtrace(arr, 100);

  backtrace_symbols_fd(arr, depth, STDERR_FILENO);
}

}  // namespace chess
