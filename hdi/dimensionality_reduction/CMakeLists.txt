cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)
file(GLOB GpgpuHeaderFiles gpgpu_sne/*.h)
file(GLOB GpgpuSourceFiles  gpgpu_sne/*.cpp)

add_library (hdidimensionalityreduction 
  ${HeaderFiles} ${SourceFiles} ${GpgpuHeaderFiles} ${GpgpuSourceFiles})
target_link_libraries (hdidimensionalityreduction hdiutils hdidata flann::flann)

########### INSTALL ##############
install (TARGETS hdidimensionalityreduction 
  DESTINATION bin)
install (FILES ${HeaderFiles} 
  DESTINATION include/hdi/dimensionality_reduction)
install (FILES ${GpgpuHeaderFiles} 
  DESTINATION include/hdi/dimensionality_reduction/gpgpu_sne)
