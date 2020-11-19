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

  # Copy over libopenblas shared libraries for FAISS dependency (ugh) 
  if (MSVC) 
    add_custom_command(TARGET ${Name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/third_party/openblas/Win/bin
        $<TARGET_FILE_DIR:${Name}>)
  endif()
endfunction()

addProject(vec_to_bin vec_to_bin.cpp)
addProject(bin_to_txt bin_to_txt.cpp)
addProject(renderer_cmd renderer_cmd.cpp)
addProject(tsne_cmd tsne_cmd.cpp)
addProject(simulation_generator_cmd simulation_generator_cmd.cpp)
addProject(evaluation_cmd evaluation_cmd.cpp)