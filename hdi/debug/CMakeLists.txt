cmake_minimum_required (VERSION 3.10)

file(GLOB headers 
  *.hpp 
  renderer/*.hpp
  utils/*.hpp
)

file(GLOB sources 
  *.cpp 
  renderer/*.cpp
  utils/*.cpp
)

add_library(hdidebug ${headers} ${sources})
target_include_directories(hdidebug PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(hdidebug PUBLIC cxx_std_17)
target_link_libraries(hdidebug 
  hdiutils 
  hdidimensionalityreduction 
  glm 
  glfw
  imgui
)

install (TARGETS hdidebug DESTINATION bin)
install (FILES ${headers} DESTINATION include/hdi/debug)