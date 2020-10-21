# Source: https://github.com/PointCloudLibrary/pcl/blob/master/cmake/Modules/FindFLANN.cmake

# Software License Agreement (BSD License)

# Point Cloud Library (PCL) - www.pointclouds.org
# Copyright (c) 2009-2012, Willow Garage, Inc.
# Copyright (c) 2012-, Open Perception, Inc.
# Copyright (c) XXX, respective authors.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: 

#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#.rst:
# FindFLANN
# --------
#
# Try to find FLANN library and headers. This module supports both old released versions
# of FLANN ≤ 1.9.1 and newer development versions that ship with a modern config file.
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` targets:
#
# ``flann::flann``
#  Defined if the system has FLANN.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module sets the following variables:
#
# ::
#
#   FLANN_FOUND               True in case FLANN is found, otherwise false
#   FLANN_ROOT                Path to the root of found FLANN installation
#
# Example usage
# ^^^^^^^^^^^^^
#
# ::
#
#     find_package(FLANN REQUIRED)
#
#     add_executable(foo foo.cc)
#     target_link_libraries(foo flann::flann)
#

# Early return if FLANN target is already defined. This makes it safe to run
# this script multiple times.
if(TARGET flann::flann)
  return()
endif()

# First try to locate FLANN using modern config
find_package(flann NO_MODULE ${FLANN_FIND_VERSION} QUIET)
if(flann_FOUND)
  unset(flann_FOUND)
  set(FLANN_FOUND ON)
  # Create interface library that effectively becomes an alias for the appropriate (static/dynamic) imported FLANN target
  add_library(flann::flann INTERFACE IMPORTED GLOBAL)
  if(FLANN_USE_STATIC)
    set_property(TARGET flann::flann APPEND PROPERTY INTERFACE_LINK_LIBRARIES flann::flann_cpp_s)
  else()
    set_property(TARGET flann::flann APPEND PROPERTY INTERFACE_LINK_LIBRARIES flann::flann_cpp)
  endif()
  # Determine FLANN installation root based on the path to the processed Config file
  get_filename_component(_config_dir "${flann_CONFIG}" DIRECTORY)
  get_filename_component(FLANN_ROOT "${_config_dir}/../../.." ABSOLUTE)
  unset(_config_dir)
  return()
endif()

# Second try to locate FLANN using pkgconfig
find_package(PkgConfig QUIET)
if(FLANN_FIND_VERSION)
  pkg_check_modules(PC_FLANN flann>=${FLANN_FIND_VERSION})
else()
  pkg_check_modules(PC_FLANN flann)
endif()

find_path(FLANN_INCLUDE_DIR
  NAMES
    flann/flann.hpp
  HINTS
    ${PC_FLANN_INCLUDE_DIRS}
    ${FLANN_ROOT}
    $ENV{FLANN_ROOT}
  PATHS
    $ENV{PROGRAMFILES}/Flann
    $ENV{PROGRAMW6432}/Flann
  PATH_SUFFIXES
    include
)

if(FLANN_USE_STATIC)
  set(FLANN_RELEASE_NAME flann_cpp_s)
  set(FLANN_DEBUG_NAME flann_cpp_s-gd)
  set(FLANN_LIBRARY_TYPE STATIC)
else()
  set(FLANN_RELEASE_NAME flann_cpp)
  set(FLANN_DEBUG_NAME flann_cpp-gd)
  set(FLANN_LIBRARY_TYPE SHARED)
endif()

find_library(FLANN_LIBRARY
  NAMES
    ${FLANN_RELEASE_NAME}
  HINTS
    ${PC_FLANN_LIBRARY_DIRS}
    ${FLANN_ROOT}
    $ENV{FLANN_ROOT}
  PATHS
    $ENV{PROGRAMFILES}/Flann
    $ENV{PROGRAMW6432}/Flann
  PATH_SUFFIXES
    lib
)

find_library(FLANN_LIBRARY_DEBUG
  NAMES
    ${FLANN_DEBUG_NAME}
  HINTS
    ${PC_FLANN_LIBRARY_DIRS}
    ${FLANN_ROOT}
    $ENV{FLANN_ROOT}
  PATHS
    $ENV{PROGRAMFILES}/Flann
    $ENV{PROGRAMW6432}/Flann
  PATH_SUFFIXES
    lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FLANN DEFAULT_MSG
  FLANN_LIBRARY FLANN_INCLUDE_DIR
)

if(FLANN_FOUND)
  add_library(flann::flann ${FLANN_LIBRARY_TYPE} IMPORTED GLOBAL)
  set_target_properties(flann::flann PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FLANN_INCLUDE_DIR}")
  set_target_properties(flann::flann PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${PC_FLANN_CFLAGS_OTHER}")
  set_property(TARGET flann::flann APPEND PROPERTY IMPORTED_CONFIGURATIONS "RELEASE")
  set_target_properties(flann::flann PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX")
  if(WIN32 AND NOT FLANN_USE_STATIC)
    set_target_properties(flann::flann PROPERTIES IMPORTED_IMPLIB_RELEASE "${FLANN_LIBRARY}")
  else()
    set_target_properties(flann::flann PROPERTIES IMPORTED_LOCATION_RELEASE "${FLANN_LIBRARY}")
  endif()
  if(FLANN_LIBRARY_DEBUG)
    set_property(TARGET flann::flann APPEND PROPERTY IMPORTED_CONFIGURATIONS "DEBUG")
    set_target_properties(flann::flann PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX")
    if(WIN32 AND NOT FLANN_USE_STATIC)
      set_target_properties(flann::flann PROPERTIES IMPORTED_IMPLIB_DEBUG "${FLANN_LIBRARY_DEBUG}")
    else()
      set_target_properties(flann::flann PROPERTIES IMPORTED_LOCATION_DEBUG "${FLANN_LIBRARY_DEBUG}")
    endif()
  endif()
  # Pkgconfig may specify additional link libraries besides from FLANN itself
  # in PC_FLANN_LIBRARIES, add them to the target link interface.
  foreach(_library ${PC_FLANN_LIBRARIES})
    if(NOT _library MATCHES "flann")
      set_property(TARGET flann::flann APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${_library}")
    endif()
  endforeach()
  get_filename_component(FLANN_ROOT "${FLANN_INCLUDE_DIR}" PATH)
endif()