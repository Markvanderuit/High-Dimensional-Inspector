cmake_minimum_required (VERSION 3.16)

# Create a C++/CUDA project
project(HighDimInspector LANGUAGES C CXX CUDA)

# Add custom find_package modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

# Disable some MSVC warnings (yeah yeah yeah... I know)
if(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4018 /wd4244 /wd4305 /wd4267 /wd4661")
endif()

# Set default cuda architectures globally
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 52 75)
    message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Add third party libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flann)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/roaring)

# Add OpenMP support for core libraries 
find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add core libraries
add_subdirectory(hdi/data)
add_subdirectory(hdi/utils)
add_subdirectory(hdi/debug)
add_subdirectory(hdi/dimensionality_reduction)

# Add command line tools
add_subdirectory(applications)
