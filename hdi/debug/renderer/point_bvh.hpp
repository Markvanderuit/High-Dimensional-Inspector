#pragma once

#include <array>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"

namespace hdi::dbg {
  class PointBVHRenderer : private RenderComponent {
  public:
    PointBVHRenderer();
    ~PointBVHRenderer();

    void init(const dr::EmbeddingBVH<3> &bvh, GLuint boundsBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    // Enum values matching to buffers in _buffers array
    enum class BufferType {
      eCubeVertices,
      eCubeIndices,
      eFlags,
      eFocusPosition,
      
      Length
    };

    // Enum values matching to vertex arrays in _vertexArrays array
    enum class VertexArrayType {
      eCube,
      ePoints,
      eFocusPos,

      Length
    };

    // Enum values matching to shader programs in _programs array
    enum class ProgramType {
      eFlags,
      eOrderDraw,
      eBvhDraw,
      eFocusPosDraw,

      Length
    };

    size_t _nPos;
    size_t _nCube;
    GLuint _boundsBuffer;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // This is gonna bite me, but this whole thing is a hack at this point anyways 
    const dr::EmbeddingBVH<3> *_bvh;
    
    // ImGui default settings
    bool _drawCube = false;
    bool _drawBarnesHut = false;
    bool _drawOrder = false;
    float _flagTheta = 0.5f;
    float _orderLineWidth = 1.f;
    float _cubeLineWidth = 1.f;
    float _cubeTheta = 0.f;// 0.75f;
    float _orderOpacity = 0.75f;
    float _cubeOpacity = 1.0f;
    float _barnesHutOpacity = 1.0f;
  };
}