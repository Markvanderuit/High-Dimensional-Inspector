cmake_minimum_required (VERSION 3.10)

file(GLOB headers
  *.h 
  *hpp
  gpgpu_sne/*.h 
  gpgpu_sne/bvh/*.h
  gpgpu_sne/utils/*.h
  gpgpu_sne/*.hpp
  gpgpu_sne/bvh/*.hpp
  gpgpu_sne/utils/*.hpp
)

file(GLOB sources
  *.cpp
  gpgpu_sne/*.cpp 
  gpgpu_sne/bvh/*.cpp 
  gpgpu_sne/utils/*.cpp 
  gpgpu_sne/bvh/*.cu
  gpgpu_sne/bvh/utils/*.cu
)

add_library(hdidimensionalityreduction ${headers} ${sources})
target_include_directories(hdidebug PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(hdidimensionalityreduction PUBLIC cuda_std_14)
target_link_libraries(hdidimensionalityreduction
  hdiutils 
  hdidebug 
  hdidata 
  cub 
  cuda_helper 
  flann::flann 
  glm
)
target_compile_options(hdidimensionalityreduction PUBLIC 
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
    -gencode arch=compute_30,code=sm_30 
    -gencode arch=compute_52,code=sm_52 
    -gencode arch=compute_75,code=sm_75
    --use_fast_math 
    -O3
  >"
)


install (TARGETS hdidimensionalityreduction DESTINATION bin)
install (FILES ${headers} DESTINATION include/hdi/dimensionalityreduction)