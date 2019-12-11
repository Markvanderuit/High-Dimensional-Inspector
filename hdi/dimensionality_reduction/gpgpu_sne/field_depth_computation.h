#pragma once

#ifndef __APPLE

#include <iostream>
#include <QMatrix4x4>
#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include "3d_utils.h"

namespace hdi::dr {
  class DepthFieldComputation {
  private:
    enum TextureType {
      STENCIL_DEPTH,
      FIELD
    };

  public:
    DepthFieldComputation();
    ~DepthFieldComputation();

    // Create components for field texture computation
    void initialize(unsigned n);

    // Destroy components for field texture computation
    void clean();

    // Compute field and depth textures
    void compute(unsigned w, unsigned h,
                 float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff,
                 Bounds3D bounds, QMatrix4x4 view);

    GLuint stencilDepthTexture() const {
      return _textures[STENCIL_DEPTH];
    }

    GLuint fieldTexture() const {
      return _textures[FIELD];
    }

  private:
    bool _initialized;
    int _iteration;

    ShaderProgram _stencil_program;
    ShaderProgram _field_program;

    GLuint _point_vao;
    GLuint _stencil_fbo;
    GLuint _depth_buffer;
    std::array<GLuint, 3> _textures;
  };
}

#endif