cmake_minimum_required(VERSION 3.11)

include_directories ("${CMAKE_SOURCE_DIR}")

find_package(Qt5 REQUIRED Gui Core)

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library (hdidata ${HeaderFiles} ${SourceFiles} )
target_link_libraries (hdidata Qt5::Core Qt5::Gui roaring::roaring)

########### INSTALL ##############
install (TARGETS hdidata DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/data)
