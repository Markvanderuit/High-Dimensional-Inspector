#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include <glm/gtx/transform.hpp>
#include <array>
#include <iostream>
#include <sstream>

namespace hdi::dbg {
  RenderManager * RenderManager::_currentManager = nullptr;

  RenderComponent::RenderComponent(int priority, bool isInit)
  : _priority(priority), _isInit(isInit)
  {
    if (isInit) {
      init();
    }
  }
  
  RenderComponent::~RenderComponent()
  {
    if (_isInit) {
      destr();
    }
  }
  
  void RenderComponent::init()
  {
    auto *ptr = RenderManager::currentManager();
    if (ptr) {
      ptr->addRenderComponent(this);
      _isInit = true;
    }
  }

  void RenderComponent::destr()
  {
    auto *ptr = RenderManager::currentManager();
    if (_isInit && ptr) {
      ptr->removeRenderComponent(this);
    }
    _isInit = false;
  }

  void RenderComponent::render(glm::mat4 transform, glm::ivec4 viewport)
  {
    // ...
  }

  RenderManager::RenderManager(uint nDimensions)
  : _nDimensions(nDimensions),
    _components(cmpRenderComponent),
    _framebufferSize(0, 0)
  {
    _currentManager = this;
    _trackball.init();

    // Construct framebuffer and attachments
    glCreateFramebuffers(1, &_framebuffer);
    glCreateTextures(GL_TEXTURE_2D, 1, &_framebufferColorTexture);
    glCreateTextures(GL_TEXTURE_2D, 1, &_framebufferDepthTexture);
  }

  RenderManager::~RenderManager()
  {
    _currentManager = nullptr;
    _trackball.destr();

    // Destroy framebuffer and attachments
    glDeleteFramebuffers(1, &_framebuffer);
    glDeleteTextures(1, &_framebufferColorTexture);
    glDeleteTextures(1, &_framebufferDepthTexture);
  }

  void RenderManager::render()
  {
    _trackball.compute();

    // Get window pointer
    Window *window = Window::currentWindow();
    if (!window) {
      return;
    }
    
    // Get window framebuffer size
    // No need to render empty or offscreen window
    glm::ivec2 size = window->size();
    if (size == glm::ivec2(0) || !window->isVisible()) {
      return;
    }

    // Regenerate framebuffer textures if size has changed
    if (_framebufferSize != size) {
      _framebufferSize = size;
      if (glIsTexture(_framebufferColorTexture)) {
        glDeleteTextures(1, &_framebufferColorTexture);
        glDeleteTextures(1, &_framebufferDepthTexture);
      }
      glCreateTextures(GL_TEXTURE_2D, 1, &_framebufferColorTexture);
      glCreateTextures(GL_TEXTURE_2D, 1, &_framebufferDepthTexture);
      glTextureStorage2D(_framebufferColorTexture, 1, GL_RGBA32F, size.x, size.y);
      glTextureStorage2D(_framebufferDepthTexture, 1, GL_DEPTH_COMPONENT24, size.x, size.y);
      glNamedFramebufferTexture(_framebuffer, GL_COLOR_ATTACHMENT0, _framebufferColorTexture, 0);
      glNamedFramebufferTexture(_framebuffer, GL_DEPTH_ATTACHMENT, _framebufferDepthTexture, 0);
    }
    
    // Calculate transformation matrix
    glm::mat4 transform;
    if (_nDimensions > 2) {
      glm::mat4 model = glm::rotate(glm::radians(45.f), glm::vec3(0, 1, 0)) 
                      * glm::rotate(glm::radians(30.f), glm::vec3(1, 0, 1)) 
                      * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));
      glm::mat4 view = _trackball.matrix();
      glm::mat4 proj = glm::perspectiveFov(0.8f, (float) size.x, (float) size.y, 0.0001f, 1000.f);
      transform = proj * view * model;
    } else {
      transform = glm::scale(glm::vec3(1.66, 1.66, 1))
                * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));
    }
    
    glm::ivec4 viewport(0, 0, size.x, size.y);

    // Specify and clear framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    constexpr std::array<float, 4> clearColor = { 1.0f, 1.0f, 1.0f, 0.f };
    constexpr float clearDepth = 1.f;
    glClearNamedFramebufferfv(_framebuffer, GL_COLOR, 0, clearColor.data());
    glClearNamedFramebufferfv(_framebuffer, GL_DEPTH, 0, &clearDepth);

    // Specify depth test, blending, viewport size
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
    glViewport(0, 0, size.x, size.y);

    // Render components in added order into framebuffer
    for (auto *ptr : _components) {
      ptr->render(transform, viewport);
    }

    // Copy to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBlitNamedFramebuffer(_framebuffer, 0, 
      0, 0, size.x, size.y, 
      0, 0, size.x, size.y,
      GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST  
    );
  }

  void RenderManager::addRenderComponent(RenderComponent *ptr)
  {
    if (_components.find(ptr) == _components.end()) {
      _components.insert(ptr);
    }
  }

  void RenderManager::removeRenderComponent(RenderComponent *ptr)
  {
    if (_components.find(ptr) != _components.end()) {
      _components.erase(ptr);
    }
  }
  
  RenderManager *RenderManager::currentManager()
  {
    return _currentManager;
  }
};