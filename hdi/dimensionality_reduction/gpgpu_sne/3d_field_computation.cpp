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
#include <random>
#include <string>
#include <bitset>
#include "3d_field_computation.h"
#include "3d_field_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/log_helper_functions.h"

// Magic numbers
constexpr float pointSize = 2.f;

namespace hdi::dr {
  BaselineFieldComputation::BaselineFieldComputation()
  : _initialized(false), _iteration(0), _w(0), _h(0), _d(0) {
    // ...
  }

  BaselineFieldComputation::~BaselineFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  void BaselineFieldComputation::initialize(const TsneParameters& params, unsigned n) {
    TIMERS_CREATE()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_GRID].addShader(VERTEX, grid_vert_src);
      _programs[PROGRAM_GRID].addShader(FRAGMENT, grid_fragment_src);
      _programs[PROGRAM_FIELD_3D].addShader(COMPUTE, field_src); 
      _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Fill _cellData for precomputed voxel grid cell data
    std::vector<uint32_t> bitData;
    for (int i = 31; i >= 0; i--) {
      // bitData.push_back(1u << i | (1u << std::min(31, i + 1)) | (1u << std::max(0, i - 1)));
      bitData.push_back(1u << i);
    }
    _cellData.fill(0u);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < bitData.size(); j++) {
        _cellData[i * 4 * bitData.size() + i + 4 * j] = bitData[j];
      }
    }

    glGenTextures(_textures.size(), _textures.data());
    glGenFramebuffers(_framebuffers.size(), _framebuffers.data());

    // Generate voxel grid cell texture
    glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    // Generate voxel grid texture
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 3d fields texture
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Generate framebuffer for grid computation
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_GRID]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[TEXTURE_GRID], 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;
  }

  void BaselineFieldComputation::clean() {
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

  void BaselineFieldComputation::compute(unsigned w, unsigned h, unsigned d,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                Bounds3D bounds) {
    Point3D minBounds = bounds.min;
    Point3D range = bounds.range();

    // Rescale textures if necessary
    if (_w != w || _h != h || _d != d) {
      _w = w;
      _h = h;
      _d = d;

      // Update cell data for voxel grid computation only to d = 128
      glActiveTexture(GL_TEXTURE0);
      if (_d <= 128u) {
        glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI, _d, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, _cellData.data());
      }
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, _w, _h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _w, _h, _d, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    glAssert("Before field");

    // Compute grid texture
    {
      TIMER_TICK(GRID)
      auto& program = _programs[PROGRAM_GRID];
      program.bind();
      
      // Bind uniforms to program
      program.uniform1i("cellMap", 0);
      program.uniform1f("zPadding", 0.025f);

      // Bind the textures for this program
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);

      // Bind bounds buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Prepare for drawing
      glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_GRID]);
      glViewport(0, 0, _w, _h);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glClear(GL_COLOR_BUFFER_BIT);
      glEnable(GL_COLOR_LOGIC_OP);
      glLogicOp(GL_OR);
      
      // Draw a point at each embedding position
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(pointSize);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_LOGIC_OP);
      program.release();
      TIMER_TOCK(GRID)
    }

    // Compute fields 3D texture
    {
      TIMER_TICK(FIELD_3D)
      auto &program = _programs[PROGRAM_FIELD_3D];
      program.bind();

      // Attach grid texture for sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);

      // Set uniforms in compute shader
      program.uniform1ui("num_points", n);
      program.uniform1ui("grid_depth", std::min(128u, _d));
      program.uniform3ui("texture_size", _w, _h, _d);
      program.uniform1i("grid_texture", 0);

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_FIELD_3D], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(_w, _h, 2);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      
      program.release();
      TIMER_TOCK(FIELD_3D)
    }

    // Query field texture for positional values
    {
      TIMER_TICK(INTERP)
      auto &program = _programs[PROGRAM_INTERP];
      program.bind();

      // Bind the textures for this shader
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Bind uniforms for this shader
      program.uniform1ui("num_points", n);
      program.uniform3ui("texture_size", _w, _h, _d);
      program.uniform1i("fields_texture", 0);

      // Dispatch compute shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(128, 1, 1);

      // Cleanup
      program.release();
      TIMER_TOCK(INTERP)
    }

    // Test output textures every 100 iterations
    #ifdef FIELD_IMAGE_OUTPUT
    // if (_iteration % 10 == 0) {
        // ...
    // }
    #endif // FIELD_IMAGE_OUTPUT 
    
    TIMERS_UPDATE()
    if (_iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "");
      TIMER_LOG(_logger, GRID, "Grid")
      TIMER_LOG(_logger, FIELD_3D, "Field")
      TIMER_LOG(_logger, INTERP, "Interp")
      utils::secureLog(_logger, "");
    }
    _iteration++;
    glAssert("After tock");
  }
}