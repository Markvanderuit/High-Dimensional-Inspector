#pragma once

#include <array>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/field_bvh.h"

namespace hdi::dbg {
  template <unsigned D>
  class PixelBVHRenderer : private RenderComponent {
  private:
    typedef unsigned uint;

  public:
    PixelBVHRenderer();
    ~PixelBVHRenderer();

    void init(const dr::FieldBVH<3> &bvh, GLuint boundsBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    // Enum values matching to buffers in _buffers array
    enum class BufferType {
      eCubeVertices,
      eCubeIndices,
      eFlags,
      eDispatch,
      
      Length
    };

    // Enum values matching to vertex arrays in _vertexArrays array
    enum class VertexArrayType {
      eCube,
      ePoints,

      Length
    };

    // Enum values matching to shader programs in _programs array
    enum class ProgramType {
      eFlags,
      eBvhDraw,
      eFieldSlice,
      eFieldResample,
      eDispatch,

      Length
    };

    enum class TextureType {
      eField,
      eOutput,

      Length
    };

    size_t _nCube;
    GLuint _boundsBuffer;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<TextureType, GLuint> _textures;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // This is gonna bite me, but this whole thing is a hack at this point anyways 
    const dr::FieldBVH<3> *_bvh;
    
    // ImGui default settings
    bool _drawCube = false;
    bool _drawBarnesHut = false;
    bool _drawSpecificLevel = false;
    bool _drawField = true;
    uint _fieldLvl = 0;
    float _fieldDepth = 0.5f;
    ImVec2 _fieldDims = ImVec2(0, 0);
    ImVec2 _outputDims = ImVec2(0, 0);
    float _flagTheta = 0.5f;
    float _orderLineWidth = 1.f;
    float _cubeLineWidth = 1.f;
    float _cubeTheta = 0.f;// 0.75f;
    float _orderOpacity = 0.75f;
    float _cubeOpacity = 1.0f;
    float _barnesHutOpacity = 1.0f;
  };
}