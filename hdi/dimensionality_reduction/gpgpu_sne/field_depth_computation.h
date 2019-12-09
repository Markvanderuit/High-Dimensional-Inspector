#pragma once

#ifndef __APPLE

#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include <iostream>

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
    void compute(unsigned w, unsigned h, unsigned d,
                 float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff,
                 float min_x, float min_y, float min_z,
                 float max_x, float max_y, float max_z);

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