cmake_minimum_required (VERSION 3.10)

MACRO(ADD_DATA_CONVERSION_PROJECT Name Sources)
  add_executable (${Name} ${Sources})
  target_link_libraries( ${Name} 
    hdidata
    hdiutils
    hdidimensionalityreduction
    flann::flann 
    Qt5::Widgets 
  )
  MESSAGE( STATUS "Data conversion project: " ${Name} )
  set_target_properties(${Name} PROPERTIES FOLDER "Applications/Command line tools")
  target_include_directories(${Name} PRIVATE "${CMAKE_SOURCE_DIR}")
  target_include_directories(${Name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../hdi/visualization")
ENDMACRO()

file(GLOB VisualizationQObjHeaderFiles ${CMAKE_SOURCE_DIR}/hdi/visualization/*qobj.h)
file(GLOB AnalyticsQObjHeaderFiles ${CMAKE_SOURCE_DIR}/hdi/analytics/*qobj.h)
qt5_wrap_cpp(MocSources ${AnalyticsQObjHeaderFiles} ${VisualizationQObjHeaderFiles})
qt5_add_resources(Resources "../shared_resources/shared_resources.qrc")

ADD_DATA_CONVERSION_PROJECT(csv_2_bin csv_2_bin.cpp)
ADD_DATA_CONVERSION_PROJECT(multiple_csv_2_bin multiple_csv_2_bin.cpp)
ADD_DATA_CONVERSION_PROJECT(mnist_2_data mnist_2_data.cpp)
ADD_DATA_CONVERSION_PROJECT(timit_2_data timit_2_data.cpp)
ADD_DATA_CONVERSION_PROJECT(cifar_2_data cifar_2_data.cpp)
ADD_DATA_CONVERSION_PROJECT(bin_2_data bin_2_data.cpp)
ADD_DATA_CONVERSION_PROJECT(hyperspectral_image_2_hdvol hyperspectral_image_2_hdvol.cpp)
ADD_DATA_CONVERSION_PROJECT(truncate_bin truncate_bin.cpp)