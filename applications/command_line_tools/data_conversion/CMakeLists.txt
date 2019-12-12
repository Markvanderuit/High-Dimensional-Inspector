cmake_minimum_required (VERSION 3.11)

MACRO(ADD_COMMANDLINE_PROJECT Name Sources)
    add_executable (${Name} ${Sources})
    target_link_libraries (${Name} hdidimensionalityreduction)
    target_link_libraries (${Name} hdiclustering)
    target_link_libraries (${Name} hdidata)
    target_link_libraries (${Name} hdiutils)
    target_link_libraries (${Name} flann::flann)
    target_link_libraries (${Name} OpenGL::GL)

    MESSAGE( STATUS "Command line project: " ${Name} )

    set_target_properties(${Name} PROPERTIES FOLDER "Applications/Command line tools")
ENDMACRO()

########################################################################################################

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}/../../hdi/visualization")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5 REQUIRED
  Widgets
  OpenGL
)
find_package(OpenGL REQUIRED)


file(GLOB VisualizationQObjHeaderFiles ${CMAKE_SOURCE_DIR}/hdi/visualization/*qobj.h)
file(GLOB AnalyticsQObjHeaderFiles ${CMAKE_SOURCE_DIR}/hdi/analytics/*qobj.h)
qt5_wrap_cpp(MocSources ${AnalyticsQObjHeaderFiles} ${VisualizationQObjHeaderFiles})
qt5_add_resources(Resources "../shared_resources/shared_resources.qrc")

ADD_COMMANDLINE_PROJECT(csv_2_bin csv_2_bin.cpp)
ADD_COMMANDLINE_PROJECT(multiple_csv_2_bin multiple_csv_2_bin.cpp)
ADD_COMMANDLINE_PROJECT(mnist_2_data mnist_2_data.cpp)
ADD_COMMANDLINE_PROJECT(timit_2_data timit_2_data.cpp)
ADD_COMMANDLINE_PROJECT(cifar_2_data cifar_2_data.cpp)
ADD_COMMANDLINE_PROJECT(bin_2_data bin_2_data.cpp)
ADD_COMMANDLINE_PROJECT(hyperspectral_image_2_hdvol hyperspectral_image_2_hdvol.cpp)
