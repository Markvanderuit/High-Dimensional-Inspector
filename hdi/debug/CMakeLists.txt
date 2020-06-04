cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all files in "./"
file(GLOB headerFiles 
  *.hpp 
  renderer/*.hpp
  utils/*.hpp
)
file(GLOB sourceFiles 
  *.cpp 
  renderer/*.cpp
  utils/*.cpp
)

add_library(hdidebug ${headerFiles} ${sourceFiles})
set_target_properties(hdidebug PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(hdidebug 
  hdiutils 
  hdidimensionalityreduction 
  glm 
  glfw
  imgui
)

########### INSTALL ##############
install (TARGETS hdidebug DESTINATION bin)
install (FILES ${headerFiles} DESTINATION include/hdi/debug)