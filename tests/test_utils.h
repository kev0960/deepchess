#ifndef TESTS_TEST_UTILS_H
#define TESTS_TEST_UTILS_H

#include "absl/strings/str_split.h"
#include "board.h"

namespace chess {

// Notation looks like this:
// rnbqkbnr
// pppppppp
//
//
//
//
// PPPPPPPP
// RNBQKBNR
Board BoardFromNotation(std::string_view notation); 

}  // namespace chess

#endif
