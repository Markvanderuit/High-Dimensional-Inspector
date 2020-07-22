#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"

namespace hdi::dbg {
  template <unsigned D>
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
    bool _hasLabels;
    size_t _n;
    GLuint _embeddingBuffer;
    GLuint _boundsBuffer;
    GLuint _vertexArray;
    ShaderProgram _program;

    // ImGui default settings
    bool _draw = true;
    bool _drawLabels = false;
    float _pointSize = 4.f;
    float _pointOpacity = 1.f;
  };
}