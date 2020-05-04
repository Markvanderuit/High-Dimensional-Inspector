#pragma once

#include <array>
#include <glm/glm.hpp>
#include "hdi/data/shader.h"
#include "hdi/utils/abstract_log.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_utils.h"

namespace hdi::dr::bvh {
  class BVHRenderer {
    public:
      BVHRenderer();
      ~BVHRenderer();

      void init(const bvh::BVHLayout& bvhLayout, GLuint posBuffer);
      void destr();
      void compute(const bvh::BVH& bvh, GLuint posBuffer, GLuint boundsBuffer);

      GLuint texture() const {
        // return _textures[TEXR_FB_COLOR];
      }

      void setLogger(utils::AbstractLog* logger) {
        _logger = logger; 
      }

    private:
      enum BufferType {
        // Enum values matching to buffers in _buffers array
        BUFF_LINES,
        BUFF_TEST,

        // Static enum length
        BufferTypeLength
      };

      enum VertexArrayType {
        // Enum values matching to vertex arrays in _vertexArrays array
        VRAO_NODES,
        VRAO_POINTS,

        // Static enum length
        VertexArrayTypeLength
      };

      enum ProgramType {
        // Enum values matching to shader programs in _programs array
        PROG_TRIANGULATE,
        PROG_BVH_DRAW,
        PROG_EMB_DRAW,

        // Static enum length
        ProgramTypeLength
      };

    bool _initialized;
    GLuint fbo; // framebuffer object
    GLuint vao; // vertex array object
    glm::mat4 transform;
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<GLuint, VertexArrayTypeLength> _vertexArrays;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    utils::AbstractLog* _logger;
  };
}