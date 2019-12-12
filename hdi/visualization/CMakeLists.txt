cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}")
MESSAGE( STATUS "${CMAKE_CURRENT_BINARY_DIR}")

find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB QOBJHeaderFiles *_qobj.h)
file(GLOB SourceFiles *.cpp)
file(GLOB FormFiles *.ui)
file(GLOB ResourceFiles *.qrc)

qt5_wrap_ui(UiHeaders ${FormFiles})
qt5_wrap_cpp(MocSources ${QOBJHeaderFiles})
qt5_add_resources(Resources ${ResourceFiles})

add_library(hdivisualization STATIC ${HeaderFiles} ${SourceFiles} ${UiHeaders} ${MocSources} ${Resources})

SET_TARGET_PROPERTIES(hdivisualization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries (hdivisualization Qt5::Widgets)
target_link_libraries (hdivisualization Qt5::OpenGL)
target_link_libraries (hdivisualization hdiutils)

########### INSTALL ##############
install (TARGETS hdivisualization DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/visualization)
