if(MSVC)
  add_library(flann::flann_import STATIC IMPORTED)
  set_target_properties(flann::flann_import PROPERTIES
    IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/Win/Debug/flann_cpp_s.lib
    IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/Win/Release/flann_cpp_s.lib
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Win/Include
  )
  add_library(flann::lz4_import STATIC IMPORTED)
  set_target_properties(flann::lz4_import PROPERTIES
    IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/Win/Debug/lz4_s.lib
    IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/Win/Release/lz4_s.lib
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Win/Include
  )
  add_library(flann_lz4_interface INTERFACE)
  target_link_libraries(flann_lz4_interface INTERFACE flann::flann_import flann::lz4_import)
  add_library(flann::flann ALIAS flann_lz4_interface)
else()
  find_package(FLANN REQUIRED)
endif()