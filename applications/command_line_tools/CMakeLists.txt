cmake_minimum_required (VERSION 3.10)

MACRO(ADD_PROJECT Name Sources MocSources)
  add_executable (${Name} ${Sources} ${MocSources})

  # target_link_libraries (${Name} hdianalytics)
  target_link_libraries (${Name} hdidimensionalityreduction)
  target_link_libraries (${Name} hdiclustering)
  target_link_libraries (${Name} hdidata)
  target_link_libraries (${Name} hdivisualization)
  target_link_libraries (${Name} hdiutils)
  target_link_libraries (${Name} flann::flann)
  target_link_libraries (${Name} Qt5::Widgets)
  target_link_libraries (${Name} Qt5::OpenGL)
  target_link_libraries (${Name} OpenGL::GL)

  MESSAGE( STATUS "Command line project: " ${Name} )

  set_target_properties(${Name} PROPERTIES FOLDER "Applications/Command line tools")
ENDMACRO()

MACRO(ADD_COMMANDLINE_PROJECT Name Sources)
  add_executable (${Name} ${Sources})
  target_link_libraries (${Name} hdidimensionalityreduction)
  target_link_libraries (${Name} hdiclustering)
  target_link_libraries (${Name} hdidata)
  target_link_libraries (${Name} hdiutils)
  target_link_libraries (${Name} flann::flann)

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
qt5_wrap_cpp(MocSources ${VisualizationQObjHeaderFiles})
qt5_add_resources(Resources "../shared_resources/shared_resources.qrc")

ADD_PROJECT(atsne_pva_cmd atsne_pva_cmd.cpp "${Resources}")
# ADD_PROJECT(hsne_cmd hsne_cmd.cpp "${Resources}")
ADD_PROJECT(simple_embedding_viewer simple_embedding_viewer.cpp "${Resources}")
ADD_PROJECT(3d_embedding_viewer 3d_embedding_viewer.cpp "${Resources}")
ADD_PROJECT(pex_viewer pex_viewer.cpp  "${Resources}")
ADD_PROJECT(gpgpusne_cmd gpgpusne_cmd.cpp  "${Resources}")
ADD_COMMANDLINE_PROJECT(tsne_cmd tsne_cmd.cpp)
ADD_COMMANDLINE_PROJECT(atsne_cmd atsne_cmd.cpp)
ADD_COMMANDLINE_PROJECT(tsne_distance_matrix_cmd tsne_distance_matrix_cmd.cpp)