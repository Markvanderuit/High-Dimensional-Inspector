#pragma once

#include <set>
#include <glad/glad.h>
#include "hdi/debug/utils/trackball.hpp"

namespace hdi::dbg {
  class RenderComponent {
  public:
    RenderComponent(int priority = 0, bool isInit = false);
    ~RenderComponent();

    // Lifetime management
    virtual void init();
    virtual void destr();

    // Actual rendering
    virtual void render(glm::mat4 transform, glm::ivec4 viewport);
    
    // For sorted set insertion
    friend bool cmpRenderComponent(RenderComponent *a, RenderComponent *b) {
      return a->_priority < b->_priority && a != b;
    };
  private:
    int _priority;
    bool _isInit;
  };

  class RenderManager {
  public:
    RenderManager();
    ~RenderManager();
    
    // Render all composite components
    void render();

    // Register render components
    void addRenderComponent(RenderComponent *ptr);
    void removeRenderComponent(RenderComponent *ptr);

    // Static access
    static RenderManager *currentManager();

  private:
    Trackball _trackball;
    GLuint _framebuffer;
    GLuint _framebufferColorTexture;
    GLuint _framebufferDepthTexture;
    glm::ivec2 _framebufferSize;

    std::set<RenderComponent *, decltype(cmpRenderComponent)*> _components;

    static RenderManager *_currentManager;
  };
}