cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library (hdiclustering ${HeaderFiles} ${SourceFiles} )
target_link_libraries(hdiclustering flann::flann)

########### INSTALL ##############
install (TARGETS hdiclustering DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/clustering)