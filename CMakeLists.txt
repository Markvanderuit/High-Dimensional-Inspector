cmake_minimum_required (VERSION 3.10)

# Create a C++ 17 project
project(HighDimInspector LANGUAGES CXX CUDA C)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define available options
option(HDI_BUILD_VISUALIZATION "Build the visualization library" ON)
option(HDI_BUILD_CLI "Build the command line tools" ON)

# CRoaring
# Create lib from amalganation files
add_library(roaring STATIC
  ${PROJECT_SOURCE_DIR}/lib/CRoaringAmalganation/src/roaring.c
)
target_include_directories(roaring PUBLIC
  ${PROJECT_SOURCE_DIR}/lib/CRoaringAmalganation/include
)

# FLANN
# Leverage find_package on Mac/Linux/Unix
# Use included lib on Windows
if(MSVC)
  add_library(flann::flann_import STATIC IMPORTED)
  set_target_properties(flann::flann_import PROPERTIES
    IMPORTED_LOCATION_DEBUG ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Debug/flann_cpp_s.lib
    IMPORTED_LOCATION_RELEASE ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Release/flann_cpp_s.lib
    INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Include
  )
  add_library(flann::lz4_import STATIC IMPORTED)
  set_target_properties(flann::lz4_import PROPERTIES
    IMPORTED_LOCATION_DEBUG ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Debug/lz4_s.lib
    IMPORTED_LOCATION_RELEASE ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Release/lz4_s.lib
    INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/lib/Flann/Win/Include
  )
  add_library(flann_lz4_interface INTERFACE)
  target_link_libraries(flann_lz4_interface INTERFACE flann::flann_import flann::lz4_import)
  add_library(flann::flann ALIAS flann_lz4_interface)
else()
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
  find_package(FLANN REQUIRED)
endif()

# OpenMP
# Add support, if it is available
find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GLFW
# Add support for GLFW, if necessary. Docs/tests/examples are trimmed from src
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/Glfw)

# GLM
# Add glm library (is only an alias)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/glm)

# CUB
# Add CUB library, for CUDA radix sort implementation
add_library(cub INTERFACE)
target_include_directories(cub INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/cub/include)

# CUDA helper
# Add CUDA helper headers, for basic math functions
add_library(cuda_helper INTERFACE)
target_include_directories(cuda_helper INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/cuda/include)

# QT5
# Leverage find_package to find required components
find_package(Qt5 REQUIRED Core Widgets)
if(HDI_BUILD_VISUALIZATION OR HDI_BUILD_CLI)
  find_package(Qt5 REQUIRED Gui OpenGL)
endif()

# MSVC
# Disable some warnings GCC/Clang don't usually throw (yeah yeah I know)
if(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
endif()

# Build core libraries
add_subdirectory (hdi/data)
add_subdirectory (hdi/utils)
add_subdirectory (hdi/dimensionality_reduction)

# Build visualization library, requires QT
if(HDI_BUILD_VISUALIZATION)
  add_subdirectory (hdi/visualization)
endif()

# Build command line tools, requires QT and visualization toolkit
if(HDI_BUILD_CLI)
    add_subdirectory (applications/command_line_tools)
    add_subdirectory (applications/command_line_tools/data_conversion)
endif()

# Install
# We don't want to install in the default location, just use ./bin for now
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH "..." FORCE)
endif()
