#pragma once

#include <array>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/gl_bvh.h"

namespace hdi::dbg {
  class BvhRenderer : private RenderComponent {
  public:
    BvhRenderer();
    ~BvhRenderer();

    void init(const dr::BVH<3> &bvh, GLuint boundsBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    enum BufferType {
        // Enum values matching to buffers in _buffers array
        BUFF_CUBE_VERT,
        BUFF_CUBE_IDX,
        BUFF_FLAGS,
        BUFF_FOCUS_POS,
        BUFF_FOCUS_LINES,

        // Static enum length
        BufferTypeLength
      };

    enum VertexArrayType {
      // Enum values matching to vertex arrays in _vertexArrays array
      VRAO_CUBE,
      VRAO_POINTS,
      VRAO_FOCUS_POS,
      VRAO_FOCUS_LINES,

      // Static enum length
      VertexArrayTypeLength
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROG_FLAGS,
      PROG_ORDER_DRAW,
      PROG_BVH_INSTANCED_DRAW,
      PROG_FOCUS_POS_DRAW,

      // Static enum length
      ProgramTypeLength
    };

    size_t _nPos;
    size_t _nCube;
    GLuint _boundsBuffer;
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<GLuint, VertexArrayTypeLength> _vertexArrays;
    std::array<ShaderProgram, ProgramTypeLength> _programs;

    // This is gonna bite me, but this whole thing is a hack anyways 
    const dr::BVH<3> *_bvh;
    
    unsigned _iterations = 0;

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