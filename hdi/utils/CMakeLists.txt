cmake_minimum_required (VERSION 3.10)

file(GLOB headers 
  *.hpp 
)

file(GLOB sources 
  *.cpp 
)

add_library(hdiutils ${headers} ${sources})
target_include_directories(hdiutils PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(hdiutils PUBLIC cxx_std_17)

# ########### INSTALL ##############
install(TARGETS hdiutils DESTINATION bin)
install(FILES ${headers} DESTINATION include/hdi/utils)
