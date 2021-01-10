
add_executable(chess main.cc)
target_compile_features(chess PRIVATE cxx_std_17)

target_link_libraries(chess PRIVATE libdeepchess fmt::fmt)
