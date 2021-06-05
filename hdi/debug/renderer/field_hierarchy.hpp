#pragma once

#include <array>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy.h"

namespace hdi::dbg {
  template <unsigned D>
  class FieldHierarchyRenderer : private RenderComponent {
  private:
    using uint = unsigned;

  public:
    FieldHierarchyRenderer();
    ~FieldHierarchyRenderer();

    void init(const dr::FieldHierarchy<D> &hierarchy, GLuint boundsBuffer, GLuint pairsBuffer = 0, GLuint pairsHead = 0);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

    bool isInit() const {
      return _isInit;
    }

  private:
    // Enum values matching to buffers in _buffers array
    enum class BufferType {
      eCubeVertices,
      eCubeIndices,
      eQuadVertices,
      eQuadLineIndices,
      eQuadTriangleIndices,
      eFlags,
      
      Length
    };

    // Enum values matching to vertex arrays in _vertexArrays array
    enum class VertexArrayType {
      eQuad,
      eCube,

      Length
    };

    // Enum values matching to shader programs in _programs array
    enum class ProgramType {
      eFlags,
      eBvhDraw,
      eFieldSlice,

      Length
    };

    enum class TextureType {
      eOutput,

      Length
    };

    size_t _nCube;
    GLuint _boundsBuffer;
    GLuint _pairsBuffer;
    GLuint _pairsHead;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<TextureType, GLuint> _textures;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // This is gonna bite me, but this whole thing is a hack at this point anyways 
    const dr::FieldHierarchy<D> *_hierarchy;
    
    // ImGui default settings
    bool _drawCube = true;
    bool _drawLvl = false;
    bool _drawFlags = false;
    bool _doSum = false;
    bool _drawField = false;
    bool _drawFieldOld = false;
    uint _bvhLvl = 7;
    uint _fieldLvl = 0;
    float _fieldDepth = 0.5f;
    ImVec2 _fieldDims = ImVec2(0, 0);
    ImVec2 _outputDims = ImVec2(0, 0);
    float _cubeLineWidth = 2.f;
    float _cubeOpacity = 0.0f;
  };
}