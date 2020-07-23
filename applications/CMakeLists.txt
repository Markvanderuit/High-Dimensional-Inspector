cmake_minimum_required (VERSION 3.10)
 
function(addProject Name Sources)
  add_executable(${Name} ${Sources})
  target_compile_features(${Name} PRIVATE cxx_std_17)
  target_include_directories(${Name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${Name} 
    cxxopts
    flann::flann
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
addProject(bh_tsne_cmd bh_tsne_cmd.cpp)