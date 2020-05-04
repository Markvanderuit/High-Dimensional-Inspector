cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp) 
file(GLOB GpgpuHeaderFiles gpgpu_sne/*.h gpgpu_sne/bvh/*.h)
file(GLOB GpgpuSourceFiles gpgpu_sne/*.cpp gpgpu_sne/bvh/*.cpp gpgpu_sne/*.cu gpgpu_sne/bvh/*.cu)

add_library (hdidimensionalityreduction 
  ${HeaderFiles} 
  ${SourceFiles} 
  ${GpgpuHeaderFiles} 
  ${GpgpuSourceFiles})
# target_compile_features(hdidimensionalityreduction PUBLIC cxx_std_11)
set_target_properties(hdidimensionalityreduction PROPERTIES CUDA_STANDARD 14)
target_compile_features(hdidimensionalityreduction PUBLIC cxx_std_14)
target_link_libraries(hdidimensionalityreduction hdiutils hdidata cub cuda_helper flann::flann glm)

########### INSTALL ##############
install (TARGETS hdidimensionalityreduction 
  DESTINATION bin)
install (FILES ${HeaderFiles} 
  DESTINATION include/hdi/dimensionality_reduction)
install (FILES ${GpgpuHeaderFiles} 
  DESTINATION include/hdi/dimensionality_reduction/gpgpu_sne)
