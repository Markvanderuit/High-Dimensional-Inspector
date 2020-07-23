cmake_minimum_required (VERSION 3.10)

function(addDataConvProject Name Sources)
  add_executable (${Name} ${Sources})
  target_compile_features(${Name} PRIVATE cxx_std_17)
  target_include_directories(${Name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries( ${Name} 
    Qt5::Widgets 
    flann::flann 
    hdidata
    hdiutils
    hdidimensionalityreduction
  )
  if (OpenMP_CXX_FOUND) 
    target_link_libraries(${Name} OpenMP::OpenMP_CXX)
  endif()
endFunction()

addDataConvProject(csv_2_bin csv_2_bin.cpp)
addDataConvProject(multiple_csv_2_bin multiple_csv_2_bin.cpp)
addDataConvProject(mnist_2_data mnist_2_data.cpp)
addDataConvProject(timit_2_data timit_2_data.cpp)
addDataConvProject(cifar_2_data cifar_2_data.cpp)
addDataConvProject(bin_2_data bin_2_data.cpp)
addDataConvProject(hyperspectral_image_2_hdvol hyperspectral_image_2_hdvol.cpp)
addDataConvProject(truncate_bin truncate_bin.cpp)