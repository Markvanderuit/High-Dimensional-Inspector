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
target_link_libraries (hdiutils
  Qt5::Core     # AAAARGH
  Qt5::Widgets  # NOOOOOO
  roaring
)

# ########### INSTALL ##############
install(TARGETS hdiutils DESTINATION bin)
install(FILES ${headers} DESTINATION include/hdi/utils)
