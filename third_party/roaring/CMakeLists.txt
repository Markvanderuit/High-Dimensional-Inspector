cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_library(roaring STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/roaring.c)
target_include_directories(roaring PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)