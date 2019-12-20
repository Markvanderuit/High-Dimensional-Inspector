#pragma once

#ifndef __APPLE

#include <array>
#include <iostream>
#include <QMatrix4x4>
#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include "3d_utils.h"

namespace hdi::dr {
  class DepthFieldComputation {
  private:
    enum TextureType {
      // Enum values
      TEXTURE_DEPTH,
      TEXTURE_FIELD_3D,

      // Static enum length
      TextureTypeLength 
    };

    enum ProgramType {
      // Enum values
      PROGRAM_DEPTH,
      PROGRAM_FIELD_3D,

      // Static enum length
      ProgramTypeLength
    };

  public:
    DepthFieldComputation();
    ~DepthFieldComputation();

    // Create components for field texture computation
    void initialize(unsigned n);

    // Destroy components for field texture computation
    void clean();

    // Compute field and depth textures
    void compute(unsigned w, unsigned h, unsigned d,
                 float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff,
                 Bounds3D bounds, QMatrix4x4 view);

    GLuint depthTexture() const {
      return _textures[TEXTURE_DEPTH];
    }

    GLuint field3DTexture() const {
      return _textures[TEXTURE_FIELD_3D];
    }

  private:
    bool _initialized;
    int _iteration;
    unsigned _w, _h, _d;

    GLuint _point_vao;
    GLuint _depth_fbo;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, TextureTypeLength> _textures;
  };
}

#endif