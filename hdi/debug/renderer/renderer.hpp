#pragma once

#include <vector>
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

  protected:
    bool _isInit;

  private:
    int _priority;
  };

  class RenderManager {
  private:
    typedef unsigned uint;
    
  public:
    RenderManager(uint nDimensions, const std::vector<uint> &labels = std::vector<uint>());
    ~RenderManager();
    
    // Render all composite components
    void render();

    // Register render components
    void addRenderComponent(RenderComponent *ptr);
    void removeRenderComponent(RenderComponent *ptr);
    const GLuint labelsBuffer() const;

    // Static access
    static RenderManager *currentManager();

  private:
    uint _nDimensions;
    Trackball _trackball;
    GLuint _labels;
    GLuint _framebuffer;
    GLuint _framebufferColorTexture;
    GLuint _framebufferDepthTexture;
    glm::ivec2 _framebufferSize;

    std::set<RenderComponent *, decltype(cmpRenderComponent)*> _components;

    static RenderManager *_currentManager;
  };
}