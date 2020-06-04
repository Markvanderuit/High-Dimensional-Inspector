#include "hdi/debug/utils/window.hpp"
#define GLFW_INCLUDE_NONE
#include "glfw/glfw3.h"
#include <glad/glad.h>
#include <algorithm>
#include <stdexcept>

namespace hdi::dbg {  
  unsigned Window::_nrHandles = 0u;
  Window * Window::_currentWindow = nullptr;

  Window::Window()
  : _isInit(false)
  { }

  Window::Window(WindowInfo windowInfo, ContextInfo contextInfo)
  : _isInit(true)
  {
    // Initialize GLFW on first window creation
    if (_nrHandles == 0u && !glfwInit()) {
      throw std::runtime_error("glfwInit() failed");
    }

    // Set context creation hints based on bit flags and version
    unsigned prf = contextInfo.profile == OpenGLProfileType::eCore
                     ? GLFW_OPENGL_CORE_PROFILE
                     : (contextInfo.profile == OpenGLProfileType::eCompatibility
                          ? GLFW_OPENGL_COMPAT_PROFILE
                          : GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, contextInfo.versionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, contextInfo.versionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, prf);

    // Set framebuffer creation hints based on bit flags
    glfwWindowHint(GLFW_SRGB_CAPABLE, (windowInfo.flags & WindowInfo::bSRGB) != 0);

    // Set window creation hints based on bit flags
    glfwWindowHint(GLFW_VISIBLE, (windowInfo.flags & WindowInfo::bOffscreen) == 0);
    glfwWindowHint(GLFW_DECORATED, (windowInfo.flags & WindowInfo::bDecorated) != 0);
    glfwWindowHint(GLFW_FLOATING, (windowInfo.flags & WindowInfo::bFloating) != 0);
    glfwWindowHint(GLFW_MAXIMIZED, (windowInfo.flags & WindowInfo::bMaximized) != 0);
    glfwWindowHint(GLFW_FOCUSED, (windowInfo.flags & WindowInfo::bFocused) != 0);
    glfwWindowHint(GLFW_RESIZABLE, (windowInfo.flags & WindowInfo::bResizable) != 0);
    
    if ((windowInfo.flags & WindowInfo::bFullscreen) != 0) {
      // Create fullscreen window
      GLFWmonitor* pMonitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* pMode = glfwGetVideoMode(pMonitor);
      glfwWindowHint(GLFW_RED_BITS, pMode->redBits);
      glfwWindowHint(GLFW_GREEN_BITS, pMode->greenBits);
      glfwWindowHint(GLFW_BLUE_BITS, pMode->blueBits);
      glfwWindowHint(GLFW_REFRESH_RATE, pMode->refreshRate);
      _handle = (void*) glfwCreateWindow(
        pMode->width, pMode->height, windowInfo.title.c_str(), pMonitor, nullptr);
    } else {
      // Create normal window
      _handle = (void*) glfwCreateWindow(
        windowInfo.width, windowInfo.height, windowInfo.title.c_str(), nullptr, nullptr);
    }

    // Check if window creation was successful
    if (!_handle) {
      throw std::runtime_error("glfwCreateWindow() failed");
    }

    // Initialize GLAD
    // Requires new context to become current
    if (_nrHandles == 0u) {
      makeCurrent();
      if (!gladLoadGL()) {
        throw std::runtime_error("gladLoadGL() failed");
      }
    }

    _nrHandles++;
  }

  Window::~Window()
  {
    if (_isInit) {
      // Check if handle should be set to nullptr
      // before destroying
      if (isCurrent()) {
        _currentWindow = nullptr;
      }

      glfwDestroyWindow((GLFWwindow*) _handle);

      // Decrement static window counter
      // and terminate GLFW on last window termination
      _nrHandles--;
      if (_nrHandles == 0u) {
        glfwTerminate();
      }
    }
  }

  Window::Window(Window&& other) noexcept
  {
    swap(other);
  }

  Window& Window::operator=(Window&& other) noexcept
  {
    swap(other);
    return *this;
  }

  void Window::swap(Window& other) noexcept
  {
    std::swap(_isInit, other._isInit);
    std::swap(_handle, other._handle);
    std::swap(_size, other._size);
    if (_currentWindow == &other) {
      _currentWindow = this;
    }
  }

  void Window::makeCurrent()
  {
    glfwMakeContextCurrent((GLFWwindow*) _handle);
    _currentWindow = this; // set to wrapper, not to window
  }

  bool Window::isCurrent() const
  {
    return ((GLFWwindow*) _handle) == glfwGetCurrentContext();
  }

  bool Window::canDisplay() const
  {
    return !glfwWindowShouldClose((GLFWwindow*) _handle);
  }

  bool Window::isVisible() const {
    return glfwGetWindowAttrib((GLFWwindow*) _handle, GLFW_VISIBLE) == GLFW_TRUE;
  }

  void Window::display()
  {
    glfwSwapBuffers((GLFWwindow*) _handle);
    glfwGetFramebufferSize((GLFWwindow*) _handle, &_size[0], &_size[1]);
  }

  void Window::processEvents()
  {
    glfwPollEvents(); // FIXME not attached to window, actually
  }

  glm::ivec2 Window::size() const
  {
    return _size;
  }

  void * Window::handle() const 
  {
    return _handle;
  }

  void Window::enableVsync(bool enabled) {
    if (!isCurrent()) {
      makeCurrent();
    }
    glfwSwapInterval(enabled ? 1 : 0);
  }

  Window * Window::currentWindow() 
  {
    return _currentWindow;
  }

  Window Window::Offscreen() 
  {
    WindowInfo info;
    info.flags = WindowInfo::bOffscreen;
    return Window(info);
  }

  Window Window::Decorated() 
  {
    WindowInfo info;
    info.flags = WindowInfo::bDecorated | WindowInfo::bFocused;
    info.width = 1024;
    info.height = 768;
    return Window(info);
  }

  Window Window::Undecorated() 
  {
    WindowInfo info;
    info.flags = WindowInfo::bFocused;
    info.width = 1024;
    info.height = 768;
    return Window(info);
  }

  Window Window::Fullscreen() 
  {
    WindowInfo info;
    info.flags = WindowInfo::bFullscreen | WindowInfo::bFocused;
    return Window(info);
  }
}