cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

find_package(Qt5 REQUIRED Gui Core)

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h gpgpu_sne/*.h)
file(GLOB SourceFiles *.cpp gpgpu_sne/*.cpp)

add_library (hdidimensionalityreduction ${HeaderFiles} ${SourceFiles} )
target_link_libraries (hdidimensionalityreduction Qt5::Core Qt5::Gui flann::flann)

########### INSTALL ##############
install (TARGETS hdidimensionalityreduction DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/dimensionality_reduction)
