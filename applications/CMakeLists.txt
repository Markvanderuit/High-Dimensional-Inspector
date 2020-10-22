cmake_minimum_required (VERSION 3.16)
 
function(addProject Name Sources)
  add_executable(${Name} ${Sources})
  target_compile_features(${Name} PRIVATE cxx_std_17)
  target_include_directories(${Name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${Name} 
    cxxopts
    hdidata
    hdiutils
    hdidebug
    hdidimensionalityreduction
  )
  if (OpenMP_CXX_FOUND) 
    target_link_libraries(${Name} OpenMP::OpenMP_CXX)
  endif()
endfunction()

addProject(gpgpu_tsne_cmd gpgpu_tsne_cmd.cpp)
addProject(simulation_generator_cmd simulation_generator_cmd.cpp)

# Copy over libopenblas shared libraries for FAISS dependency (ugh) 
if (MSVC)
  add_custom_command(TARGET gpgpu_tsne_cmd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_SOURCE_DIR}/third_party/openblas/Win/bin
      $<TARGET_FILE_DIR:gpgpu_tsne_cmd>)
endif()