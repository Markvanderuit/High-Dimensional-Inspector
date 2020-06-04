#pragma once

#include <set>
#include <string>
#include <glm/glm.hpp>

namespace hdi::dbg {
  /**
   * Possible OpenGL profile requests.
   */
  enum class OpenGLProfileType {
    eAny,          // Context does not request a specific profile
    eCore,         // Context requests a core OpenGL profile
    eCompatibility // Context requests a compatibility OpenGL profile
  };

  /**
   * Wrapper structure with necessary information for context creation.
   * Default version creates the highest available context.
   */
  struct ContextInfo {
    OpenGLProfileType profile = OpenGLProfileType::eAny;
    unsigned versionMajor = 1;
    unsigned versionMinor = 0;
  };

  /**
   * Wrapper structure with necessary information for window creation.
   * Default version creates a simple 256x256px. window
   */
  struct WindowInfo {
    enum FlagBits {
      bOffscreen = 1,  // Window is invisible for offscreen rendering
      bDecorated = 2,  // Window is decorated with title bar
      bFloating = 4,   // Window floats above all other windows
      bFullscreen = 8, // Window becomes borderless full screen
      bMaximized = 16, // Window is maximized on creation
      bFocused = 32,   // Window is focused for input on creation
      bResizable = 64, // Window is resizable
      bSRGB = 128,     // Framebuffer is sRGB capable
    };

    // Default window settings
    unsigned flags = bDecorated | bFocused;
    unsigned width = 256;
    unsigned height = 256;
    std::string title = "";
  };

  /**
   * OpenGL window wrapper, manages context, default framebuffer, 
   * swap chain, input handling and so on.
   */
  class Window {
  public:
    Window();
    Window(WindowInfo windowInfo, ContextInfo contextInfo = ContextInfo());
    ~Window();

    // Copy constr/operator is explicitly deleted
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Move constr/operator moves resource handle
    Window(Window&&) noexcept;
    Window& operator=(Window&&) noexcept;

    // Swap internals with another window object
    void swap(Window& other) noexcept;

    // Render loop
    void makeCurrent();
    void processEvents();
    void display();
    void enableVsync(bool enabled);

    // Window handling
    bool isCurrent() const;
    bool canDisplay() const;
    bool isVisible() const;
    void *handle() const;
    glm::ivec2 size() const;

    // Static access
    static Window *currentWindow();

    // Simple window constructors
    static Window Offscreen();
    static Window Decorated();
    static Window Undecorated();
    static Window Fullscreen();
    
  private:
    bool _isInit;
    void *_handle;
    glm::ivec2 _size;

    static unsigned _nrHandles;
    static Window *_currentWindow;
  };
};