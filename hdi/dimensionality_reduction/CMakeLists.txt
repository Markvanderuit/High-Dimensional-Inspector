cmake_minimum_required (VERSION 3.10)

file(GLOB headers
  *.h 
  *hpp
  gpgpu_sne/*.h 
  gpgpu_sne/sort/*.h 
  gpgpu_sne/bvh/*.h
  gpgpu_sne/utils/*.h
  gpgpu_sne/utils/cuda/*.h
  gpgpu_sne/*.hpp
  gpgpu_sne/bvh/*.hpp
)

file(GLOB sources
  *.cpp
  gpgpu_sne/*.cpp 
  gpgpu_sne/bvh/*.cpp 
  gpgpu_sne/utils/*.cpp 
  gpgpu_sne/*.cu 
  gpgpu_sne/utils/*.cu 
  gpgpu_sne/utils/cuda/*.cu 
  gpgpu_sne/bvh/*.cu
)

add_library(hdidimensionalityreduction ${headers} ${sources})
target_include_directories(hdidimensionalityreduction PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(hdidimensionalityreduction PUBLIC cuda_std_17)
target_link_libraries(hdidimensionalityreduction
  hdiutils 
  hdidebug 
  hdidata 
  cub 
  glm
  gpufaiss
  flann::flann 
)

install (TARGETS hdidimensionalityreduction DESTINATION bin)
install (FILES ${headers} DESTINATION include/hdi/dimensionalityreduction)