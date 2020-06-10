#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"

namespace hdi::dbg {
  class EmbeddingRenderer : private RenderComponent {
  public:
    EmbeddingRenderer();
    ~EmbeddingRenderer();
    
    void init(size_t n,
              GLuint embeddingBuffer,
              GLuint boundsBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    bool _isInit;
    size_t _n;
    GLuint _embeddingBuffer;
    GLuint _boundsBuffer;
    GLuint _vertexArray;
    ShaderProgram _program;

    // ImGui default settings
    bool _draw = true;
    float _pointSize = 4.f;
    float _pointOpacity = 1.f;
  };
}