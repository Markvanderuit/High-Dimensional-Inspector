cmake_minimum_required (VERSION 3.10)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}")
MESSAGE( STATUS "${CMAKE_CURRENT_BINARY_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB QOBJHeaderFiles *_qobj.h)
file(GLOB SourceFiles *.cpp)
file(GLOB FormFiles *.ui)
file(GLOB ResourceFiles *.qrc)

qt5_wrap_ui(UiHeaders ${FormFiles})
qt5_wrap_cpp(MocSources ${QOBJHeaderFiles})
qt5_add_resources(Resources ${ResourceFiles})

add_library(hdivisualization ${HeaderFiles} ${SourceFiles} ${UiHeaders} ${MocSources} ${Resources})
target_link_libraries (hdivisualization Qt5::Widgets Qt5::OpenGL hdiutils)

########### INSTALL ##############
install (TARGETS hdivisualization DESTINATION bin)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/visualization)
