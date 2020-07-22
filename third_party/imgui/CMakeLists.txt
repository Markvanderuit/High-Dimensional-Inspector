cmake_minimum_required (VERSION 3.10)

add_library(imgui
  src/imgui.cpp
  src/imgui_draw.cpp
  src/imgui_widgets.cpp
  src/imgui_impl_glfw.cpp
  src/imgui_impl_opengl3.cpp
  src/imgui_demo.cpp
)
target_include_directories(imgui PUBLIC include)
target_compile_features(imgui PUBLIC cxx_std_17)
target_link_libraries(imgui glad glfw)