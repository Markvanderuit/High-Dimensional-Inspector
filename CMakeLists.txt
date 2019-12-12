cmake_minimum_required (VERSION 3.11)

# Enforce c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (HighDimInspector LANGUAGES CXX)

# Define available options
option(HDI_USE_ROARING "Use roaring bitmaps" ON)
option(HDI_BUILD_VISUALIZATION "Build the visualization library" ON)
option(HDI_BUILD_CLI "Build command line tools" ON)

# Toggle use of roaring
if (HDI_USE_ROARING)
  find_package(roaring REQUIRED)
  add_definitions(-DPREPROC_USE_ROARING)
endif(HDI_USE_ROARING)

# Find required packages
find_package(flann REQUIRED)
find_package(OpenMP REQUIRED)

# Add OpenMP, if available
if (OPENMP_FOUND)
  message (STATUS "OpenMP found")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

# Disable some MSVC warnings GCC doesn't usually throw (yeah yeah I know)
if (MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
endif(MSVC)

# Add subdirectories where code is located
add_subdirectory (hdi/data)
add_subdirectory (hdi/utils)
# add_subdirectory (hdi/analytics)
add_subdirectory (hdi/dimensionality_reduction)
add_subdirectory (hdi/clustering)
if(HDI_BUILD_VISUALIZATION)
  add_subdirectory (hdi/visualization)
endif(HDI_BUILD_VISUALIZATION)
if(HDI_BUILD_CLI)
    add_subdirectory (applications/command_line_tools)
    add_subdirectory (applications/command_line_tools/data_conversion)
endif(HDI_BUILD_CLI)
