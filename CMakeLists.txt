# Works with 3.11 and tested through 3.18
cmake_minimum_required(VERSION 3.11...3.18)

# Project name and a few useful settings. Other commands can pick up the results
project(
  deepchess 
  VERSION 0.1
  DESCRIPTION "Deep learning based Chess engine."
  LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Formatting library
FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 7.0.3)
FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

# JSON library
FetchContent_Declare(
  json 
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.9.1)
FetchContent_MakeAvailable(json)

find_package(Torch REQUIRED PATHS ~/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Need to fix abseil linking error: Check
# https://github.com/abseil/abseil-cpp/issues/218
set(CMAKE_CXX_STANDARD 17)

# For abseil library
add_subdirectory(abseil-cpp)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(app)

option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
    add_test(chesstest chesstest)
endif()
