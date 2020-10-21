cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
