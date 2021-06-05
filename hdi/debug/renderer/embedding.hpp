#pragma once

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"

namespace hdi::dbg {
  template <unsigned D>
  class EmbeddingRenderer : public RenderComponent {
  public:
    EmbeddingRenderer();
    ~EmbeddingRenderer();
    
    void init(size_t n,
              GLuint embeddingBuffer,
              GLuint boundsBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    // Enum values matching to buffers in _buffers array
    enum class BufferType {
      eQuadVertices,
      eQuadUvs,
      eQuadIndices,
      
      Length
    };

    // Enum values matching to objects in _vertexArrays array
    enum class VertexArrayType {
      eQuad,

      Length
    };

    // Enum values matching to programs in _programs array
    enum class ProgramType {
      eQuad,

      Length
    };

    bool _hasLabels;
    size_t _n;
    GLuint _embeddingBuffer;
    GLuint _boundsBuffer;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // ImGui default settings
    bool _draw = true;
    bool _drawLabels = true;
    bool _drawGaussian = false;
    float _quadSize = 0.002f;
    float _pointOpacity = 1.0f;
  };
}