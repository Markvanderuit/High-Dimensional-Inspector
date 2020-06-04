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
#include "3d_bvh_field_computation.h"
#include "3d_bvh_field_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/log_helper_functions.h"

// Magic numbers
constexpr float pointSize = 2.f;

namespace hdi::dr {
  BVHFieldComputation::BVHFieldComputation()
  : _initialized(false), _iteration(0), _w(0), _h(0), _d(0), _logger(nullptr) {
    // ...
  }

  BVHFieldComputation::~BVHFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  void BVHFieldComputation::initialize(const TsneParameters& params, 
                                            GLuint position_buff, 
                                            unsigned n) {
    TIMERS_CREATE()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_GRID].addShader(VERTEX, grid_vert_src);
      _programs[PROGRAM_GRID].addShader(FRAGMENT, grid_fragment_src);
      _programs[PROGRAM_FIELD].addShader(COMPUTE, _field_bvh_src); 
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
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD]);
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

    _bvh.init(params, position_buff, n);

    _iteration = 0;
    _initialized = true;
  }

  void BVHFieldComputation::clean() {
    TIMERS_DESTROY()
    _bvh.destr();
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(_framebuffers.size(), _framebuffers.data());
    glDeleteVertexArrays(1, &_point_vao);
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
  }

  void BVHFieldComputation::compute(unsigned w, unsigned h, unsigned d,
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
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD]);
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _w, _h, _d, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Compute BVH structure over embedding
    _bvh.compute(bounds);

    // Compute grid texture
    {
      TIMER_TICK(TIMER_GRID)
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
      glDisable(GL_COLOR_LOGIC_OP);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      program.release();
      TIMER_TOCK(TIMER_GRID)
    }

    // Compute fields 3D texture through O(px log n) computation
    {  
      TIMER_TICK(TIMER_FIELD)
      auto &program = _programs[PROGRAM_FIELD];
      program.bind();

      // Query bvh for layout specs and buffer handle
      const auto layout = _bvh.layout();
      const GLuint buff = _bvh.buffer();

      // Compute appropriate buffer range sizes
      GLsizeiptr float4Size = 4 * sizeof(float);
      GLsizeiptr nodeSize = float4Size * (layout.nNodes + layout.nLeaves);
      GLsizeiptr massSize = sizeof(unsigned) * (layout.nNodes + layout.nLeaves);
      GLsizeiptr idxSize = sizeof(unsigned) * layout.nLeaves;
      GLsizeiptr posSize = float4Size * layout.nPos;

      // Compute appropriate buffer range offsets, accomodating 16 byte padding
      GLintptr nodeOffset = 0;
      GLintptr massOffset = nodeSize;
      GLintptr idxOffset = massOffset + ceilDiv(massSize, float4Size) * float4Size;
      GLintptr posOffset = idxOffset + ceilDiv(idxSize, float4Size) * float4Size;

      // Bind buffer as ranges
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buff, nodeOffset, nodeSize);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, buff, massOffset, massSize);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, buff, idxOffset, idxSize);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, buff, posOffset, posSize);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bounds_buff);
      
      /* if (_iteration == 0) {
        std::cout << "Pos" << std::endl;
        {
          std::vector<float> testBuffer(4 * layout.nNodes + layout.nLeaves);
          glGetNamedBufferSubData(buff, nodeOffset, nodeSize, testBuffer.data());
          for (int i = 0; i < layout.nNodes + layout.nLeaves; i++) {
            std::cout << testBuffer[4 * i] << ", "
                      << testBuffer[4 * i + 1] << ", "
                      << testBuffer[4 * i + 2] << ", "
                      << testBuffer[4 * i + 3] << '\n';
          }
        }

        std::cout << "Bounds" << std::endl;
        {
          std::vector<float> buffer(16);
          glGetNamedBufferSubData(bounds_buff, 0, 16 * sizeof(float), buffer.data());
          std::cout << "min " 
                    << buffer[0] << ", " 
                    << buffer[1] << ", "
                    << buffer[2] << '\n';
          std::cout << "max " 
                    << buffer[4] << ", " 
                    << buffer[5] << ", "
                    << buffer[6] << '\n';
        }

        // std::vector<uint> testBuffer(layout.nNodes + layout.nLeaves);
        // glGetNamedBufferSubData(buff, massOffset, massSize, testBuffer.data());
        // for (auto& v : testBuffer) {
        //   std::cout << v << '\n';
        // }

        // std::vector<uint> testBuffer(layout.nLeaves);
        // glGetNamedBufferSubData(buff, idxOffset, idxSize, testBuffer.data());
        // for (auto& v : testBuffer) {
        //   std::cout << v << '\n';
        // }

        // exit(0);
      } */

      // Bind writable image for field texture
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Bind sampler grid texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
      // glBindTextureUnit(0, _textures[TEXTURE_GRID]);

      // Set uniforms
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("kNode", layout.kNode);
      program.uniform1ui("nLvls", layout.nLvls);
      program.uniform1ui("gridDepth", std::min(128u, _d));
      program.uniform1f("theta", 0.5f);
      program.uniform3ui("textureSize", _w, _h, _d);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(_w, 4u), ceilDiv(_h, 4u), ceilDiv(_d, 4u));
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      GL_ASSERT("3dBvhFieldComputation::field");
      TIMER_TOCK(TIMER_FIELD)
    }

    // Compute fields 3D texture through O(px n) computation
    /* {
      TIMER_TICK(TIMER_FIELD)
      auto &program = _programs[PROGRAM_FIELD];
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
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(_w, _h, 2);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      
      program.release();
      TIMER_TOCK(TIMER_FIELD)
    } */

    // Query field texture for positional values
    {
      TIMER_TICK(TIMER_INTERP)
      auto &program = _programs[PROGRAM_INTERP];
      program.bind();

      // Bind the textures for this shader
      glBindTextureUnit(0, _textures[TEXTURE_FIELD]);

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Bind uniforms for this shader
      program.uniform1ui("nPoints", n);
      program.uniform3ui("fieldSize", _w, _h, _d);
      program.uniform1i("fieldSampler", 0);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(n, 128u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      // Cleanup
      program.release();
      GL_ASSERT("3dBvhFieldComputation::interp");
      TIMER_TOCK(TIMER_INTERP)
    }

    // Test output textures every 100 iterations
    #ifdef FIELD_IMAGE_OUTPUT
    if (_iteration % FIELD_IMAGE_ITERS == 0) {
      std::vector<float> buffer(4 * _w * _h * _d);
      glGetTextureImage(_textures[TEXTURE_FIELD], 0, GL_RGBA, GL_FLOAT, 4 * sizeof(float) * buffer.size(), buffer.data());
      std::vector<float> image(_w * _h);
      int z = _d / 2;
      for (int y = 0; y < _h; y++) {
        for (int x = 0; x < _w; x++) {
          int i = _w * _h * z
                + _w * y
                + x;
          image[y * _w + x] = buffer[4 * i];
        }
      }
      std::string filename = "./output/density_" + std::to_string(_iteration);
      hdi::utils::valuesToImage(filename, image, _w, _h, 1);
    }
    #endif // FIELD_IMAGE_OUTPUT 
    
    // Update timers, log values on final iteration
    TIMERS_UPDATE()
    if (_iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "Field computation");
      TIMER_LOG(_logger, TIMER_GRID, "  Grid")
      TIMER_LOG(_logger, TIMER_FIELD, "  Field")
      TIMER_LOG(_logger, TIMER_INTERP, "  Interp")
      utils::secureLog(_logger, "");
    }
    _iteration++;
  }
}