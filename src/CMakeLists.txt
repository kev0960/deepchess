include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  ${deepchess_SOURCE_DIR}/src/*.h
  ${deepchess_SOURCE_DIR}/src/*.cc
)

add_library(libdeepchess STATIC ${SRC_FILES})

target_link_libraries(libdeepchess
  fmt::fmt
  pthread
  nlohmann_json::nlohmann_json
  stdc++fs
  "${TORCH_LIBRARIES}"
  absl::strings
  zmq
  )

target_include_directories(libdeepchess PUBLIC .)
target_compile_features(libdeepchess PUBLIC cxx_std_17)

if (MSVC)
  target_compile_options(libdeepchess PRIVATE /W4 /WX)
else()
  target_compile_options(libdeepchess PRIVATE -Wall -Wextra -pedantic -Werror)
endif()
