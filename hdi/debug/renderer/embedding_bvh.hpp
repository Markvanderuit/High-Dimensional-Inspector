#pragma once

#include <array>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"

namespace hdi::dbg {
  template <unsigned D>
  class EmbeddingBVHRenderer : public RenderComponent {
  private:
    using uint = unsigned;

  public:
    EmbeddingBVHRenderer();
    ~EmbeddingBVHRenderer();

    void init(const dr::EmbeddingBVH<D> &bvh, GLuint boundsBuffer, GLuint pairsBuffer = 0, GLuint pairsHead = 0);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    // Enum values matching to buffers in _buffers array
    enum class BufferType {
      eCubeVertices,
      eCubeIndices,
      eQuadVertices,
      eQuadIndices,
      eFlags,
      eFocusPosition,
      
      Length
    };

    // Enum values matching to vertex arrays in _vertexArrays array
    enum class VertexArrayType {
      eCube,
      eQuad,
      ePoints,
      eFocusPos,

      Length
    };

    // Enum values matching to shader programs in _programs array
    enum class ProgramType {
      eFlags,
      eBvhDraw,
      eFocusPosDraw,

      Length
    };

    size_t _nPos;
    size_t _nCube;
    GLuint _boundsBuffer;
    GLuint _pairsBuffer;
    GLuint _pairsHead;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // This is gonna bite me, but this whole thing is a hack at this point anyways 
    const dr::EmbeddingBVH<D> *_bvh;
    
    // ImGui default settings
    bool _drawCube = true;
    bool _drawLvl = false;
    bool _drawFlags = false;
    uint _bvhLvl = 1;
    float _cubeLineWidth = 1.f;
    float _cubeOpacity = 1.0f;
  };
}