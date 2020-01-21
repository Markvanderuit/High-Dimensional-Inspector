/*
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include "2d_field_computation.h"
#include "2d_field_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/log_helper_functions.h"

// Magic numbers
constexpr float pointSize = 3.f;

namespace hdi::dr {
  Baseline2dFieldComputation::Baseline2dFieldComputation()
  : _initialized(false), _iteration(0), _w(0), _h(0) {
    // ...
  }

  Baseline2dFieldComputation::~Baseline2dFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  // Initialize gpu components for computation
  void Baseline2dFieldComputation::initialize(const TsneParameters& params,
                                              unsigned n) {
    TIMERS_CREATE()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_STENCIL].addShader(VERTEX, stencil_vert_src);
      _programs[PROGRAM_STENCIL].addShader(FRAGMENT, stencil_fragment_src);
      _programs[PROGRAM_FIELD].addShader(COMPUTE, field_src); 
      _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }
    
    glGenTextures(_textures.size(), _textures.data());
    glGenFramebuffers(_framebuffers.size(), _framebuffers.data());
    
    // Generate stencil texture
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_STENCIL]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate fields texture
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    
    // Generate framebuffer for grid computation
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_STENCIL]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[TEXTURE_STENCIL], 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;

    glAssert("2dFieldComputation initialize");
  }

  // Remove gpu components
  void Baseline2dFieldComputation::clean() {
    TIMERS_DESTROY()
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(_framebuffers.size(), _framebuffers.data());
    glDeleteVertexArrays(1, &_point_vao);
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
  }

  // Compute field and depth textures
  void Baseline2dFieldComputation::compute(unsigned w, unsigned h,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                Bounds2D bounds) {
    Point2D minBounds = bounds.min;
    Point2D range = bounds.range();

    // Rescale textures if necessary
    if (_w != w || _h != h) {
      _w = w;
      _h = h;

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_STENCIL]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, _w, _h, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Compute stencil texture
    {
      TIMER_TICK(TIMER_STENCIL)
      auto& program = _programs[PROGRAM_STENCIL];
      program.bind();

      // Bind bounds buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);

      // Prepare for drawing
      glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_STENCIL]);
      glViewport(0, 0, _w, _h);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glClear(GL_COLOR_BUFFER_BIT);
      
      // Draw a point at each embedding position
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(pointSize);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      program.release();
      TIMER_TOCK(TIMER_STENCIL)
    }

    // Compute fields texture
    {
      TIMER_TICK(TIMER_FIELD)
      auto& program = _programs[PROGRAM_FIELD];
      program.bind();

      // Attach stencil texture for sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_STENCIL]);

      // Set uniforms in compute shader
      program.uniform1ui("num_points", n);
      program.uniform2ui("texture_size", _w, _h);
      program.uniform1i("stencil_texture", 0);

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(_w, _h, 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      TIMER_TOCK(TIMER_FIELD)
    }

    // Query field texture for positional values
    {
      TIMER_TICK(TIMER_INTERP)
      auto& program = _programs[PROGRAM_INTERP];
      program.bind();

      // Bind the textures for this shader
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Bind uniforms for this shader
      program.uniform1ui("num_points", n);
      program.uniform2ui("texture_size", _w, _h);
      program.uniform1i("fields_texture", 0);

      // Dispatch compute shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(128, 1, 1);

      // Cleanup
      program.release();
      TIMER_TOCK(TIMER_INTERP)
    }
    
    // Update timers, log values on final iteration
    TIMERS_UPDATE()
    if (_iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "\nField computation");
      TIMER_LOG(_logger, TIMER_STENCIL, "  Stencil")
      TIMER_LOG(_logger, TIMER_FIELD, "  Field")
      TIMER_LOG(_logger, TIMER_INTERP, "  Interp")
      utils::secureLog(_logger, "");
    }
    _iteration++;
  }
}