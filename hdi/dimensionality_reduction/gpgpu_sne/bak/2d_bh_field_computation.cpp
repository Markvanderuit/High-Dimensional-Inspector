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
#include <algorithm>
#include <execution>
#include <cmath>
#include "2d_bh_field_computation.h"
#include "2d_bh_field_shaders.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/visual_utils.h"

// Magic numbers
constexpr float pointSize = 3.f;
constexpr float theta = 0.5f; // TODO should be program parameter

namespace hdi::dr {
  BarnesHut2dFieldComputation::BarnesHut2dFieldComputation()
  : _initialized(false), _w(0), _h(0) {
    // ...
  }

  BarnesHut2dFieldComputation::~BarnesHut2dFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  // Initialize gpu components for computation
  void BarnesHut2dFieldComputation::initialize(const TsneParameters& params,
                                              unsigned n) {
    INIT_TIMERS()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_STENCIL].addShader(COMPUTE, stencil_src);
      _programs[PROGRAM_TRAVERSE].addShader(COMPUTE, tree_traverse_src);
      _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }
    
    glGenTextures(_textures.size(), _textures.data());
    
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

    _densityComputation.initialize(_params, n);

    _initialized = true;
    ASSERT_GL("2dFieldComputation::initialize");
  }

  // Remove gpu components
  void BarnesHut2dFieldComputation::clean() {
    DSTR_TIMERS()
    glDeleteTextures(_textures.size(), _textures.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _densityComputation.clean();
    _initialized = false;
  }

  // Compute field and depth textures
  void BarnesHut2dFieldComputation::compute(unsigned w, unsigned h,
                float function_support, unsigned iteration, unsigned n,
                GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                Bounds2D bounds) {
    Point2D minBounds = bounds.min;
    Point2D range = bounds.range();

    // Compute density texture
    _densityComputation.compute(w, h, n,
                                position_buff, bounds_buff,
                                bounds);

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
  
    // Compute stencil texture from density texture
    {
      TICK_TIMER(TIMER_STENCIL)
      auto& program = _programs[PROGRAM_STENCIL];
      program.bind();

      // Set uniforms in compute shader
      program.uniform1i("densityAtlasSampler", 0);
      program.uniform1ui("level", _densityComputation.getDensityAtlasBaseLevel());

      // Bind density atlas to 0 for reading
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _densityComputation.getDensityAtlasTexture());

      // Bind atlas data buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _densityComputation.getDensityAtlasBuffer());

      // Bind stencil texture for writing
      glBindImageTexture(0, _textures[TEXTURE_STENCIL], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8UI);

      // Perform computation
      glDispatchCompute(ceilDiv(_w, 16u), ceilDiv(_h, 16u), 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      TOCK_TIMER(TIMER_STENCIL)
      ASSERT_GL("Density2dFieldComputation::compute::stencil")
    }

    // Compute field texture throough tree traversal
    {
      TICK_TIMER(TIMER_TRAVERSE)
      auto& program = _programs[PROGRAM_TRAVERSE];
      program.bind();

      // Attach textures for sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_STENCIL]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, _densityComputation.getDensityAtlasTexture());

      // Set uniforms
      program.uniform1ui("nPoints", n);
      program.uniform1ui("nLevels", _densityComputation.getDensityAtlasLevels().size());
      program.uniform1f("theta2", theta * theta);
      program.uniform2i("fieldTextureSize", _w, _h);
      Point2Dui atlasDims = _densityComputation.getDensityAtlasSize();
      program.uniform2i("densityAtlasSize", atlasDims.x, atlasDims.y);
      program.uniform1i("stencilSampler", 0);
      program.uniform1i("densityAtlasSampler", 1);

      // Bind images and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _densityComputation.getDensityAtlasBuffer());
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(ceilDiv(_w, 16u), ceilDiv(_h, 16u), 1u);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      TOCK_TIMER(TIMER_TRAVERSE)
      ASSERT_GL("Density2dFieldComputation::compute::traverse")
    }

    // Query field texture for positional values
    {
      TICK_TIMER(TIMER_INTERP)
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
      glDispatchCompute(ceilDiv(n, 128u), 1, 1);

      // Cleanup
      program.release();
      TOCK_TIMER(TIMER_INTERP)
      ASSERT_GL("Density2dFieldComputation::compute::interp")
    }

    // TODO REMOVE DEBUG OUTPUT
    // Perform test texture output
    /* if (iteration % 100 == 0) {
      glActiveTexture(GL_TEXTURE0);
      const Point2Dui dims = _densityComputation.getDensityAtlasSize();
      const auto& levels = _densityComputation.getDensityAtlasLevels();

      // Output tree
      {
        const uint32_t nPixels = dims.x * dims.y;
        std::vector<float> _buffer(4 * nPixels);
        glBindTexture(GL_TEXTURE_2D, _densityComputation.getDensityAtlasTexture());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, _buffer.data());
        std::vector<float> buffer(nPixels);
        #pragma omp parallel for
        for (int i = 0; i < nPixels; i++) {
          buffer[i] = _buffer[4 * i];
        }
        for (const auto& lvl : levels) {
          float maxf = 0.f;
          for (int y = lvl.offset.y; y < lvl.offset.y + lvl.size.y; y++) {
            for (int x = lvl.offset.x; x < lvl.offset.x + lvl.size.x; x++) {
              const int i = y * dims.x + x;
              maxf = std::max(maxf, buffer[i]);
            }
          }
          std::cout << "max was " << maxf << std::endl;
          maxf = 1.f;// / maxf;
          #pragma omp parallel for
          for (int y = lvl.offset.y; y < lvl.offset.y + lvl.size.y; y++) {
            for (int x = lvl.offset.x; x < lvl.offset.x + lvl.size.x; x++) {
              const int i = y * dims.x + x;
              buffer[i] = buffer[i] * maxf;
            }
          }
        }
        utils::valuesToImage("./images/tree_" + std::to_string(iteration), buffer, dims.x, dims.y, 1);
      }

      // Output field texture
      {
        std::vector<float> _buffer(4 * _w * _h, 0.f);
        std::vector<float> density(_w * _h, 0.f);
        std::vector<float> gradient(2 * _w * _h, 0.f);
        glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, _buffer.data());
        for (unsigned y = 0; y < _h; y++) {
          for (unsigned x = 0; x < _w; x++) {
            density[y * _w + x] = _buffer[4 * y * _w + 4 * x];
            gradient[2 * y * _w + 2 * x] = _buffer[4 * y * _w + 4 * x + 1];
            gradient[2 * y * _w + 2 * x + 1] = _buffer[4 * y * _w + 4 * x + 2];
          } 
        }
        auto dMaxf = *std::max_element(density.begin(), density.end());
        auto dMinf = *std::min_element(density.begin(), density.end());
        auto dRangef = 1.f / (dMaxf - dMinf);
        std::transform(std::execution::par_unseq, density.begin(), density.end(), density.begin(), 
        [dMaxf, dMinf, dRangef] (const auto& f) {
          return (f - dMinf) * dRangef;
        });
        utils::valuesToImage("./images/field_density_" + std::to_string(iteration), density, _w, _h, 1);
        auto gMaxf = *std::max_element(gradient.begin(), gradient.end());
        auto gMinf = *std::min_element(gradient.begin(), gradient.end());
        auto gRangef = 1.f / (gMaxf - gMinf);
        std::transform(std::execution::par_unseq, gradient.begin(), gradient.end(), gradient.begin(), 
        [gMaxf, gMinf, gRangef] (const auto& f) {
          return 255.f * ((f - gMinf) * gRangef);
        });
        utils::valuesToImage("./images/field_gradient_" + std::to_string(iteration), gradient, _w, _h, 2);
      }
    } */

    // Update timers, log values on final iteration
    POLL_TIMERS()
    if (iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "\nField computation");
      LOG_TIMER(_logger, TIMER_STENCIL, "  Stencil")
      LOG_TIMER(_logger, TIMER_TRAVERSE, "  Travert")
      LOG_TIMER(_logger, TIMER_INTERP, "  Interp")
      utils::secureLog(_logger, "");
    }
  }
}