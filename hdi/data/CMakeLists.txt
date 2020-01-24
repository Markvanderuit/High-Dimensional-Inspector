cmake_minimum_required(VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library (hdidata ${HeaderFiles} ${SourceFiles} )
target_link_libraries (hdidata roaring)

########### INSTALL ##############
install (TARGETS hdidata DESTINATION bin)
install (FILES ${HeaderFiles} DESTINATION include/hdi/data)
