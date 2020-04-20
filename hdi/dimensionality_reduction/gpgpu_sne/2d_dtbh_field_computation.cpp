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
#include "2d_dtbh_field_computation.h"
#include "2d_dtbh_field_shaders.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/visualization/opengl_helpers.h"
#include <algorithm>
#include <cmath>
#include <execution>

// Magic numbers
constexpr unsigned fieldTopLevel = 5; // 5 seems good
constexpr float theta = 0.5f;         // TODO extract to program parameter
constexpr unsigned nArrayLevelsRoot = 1;
constexpr unsigned nArrayLevelsTravert = 1;
constexpr unsigned nArrayLevels = std::max(nArrayLevelsRoot, nArrayLevelsTravert);

// Nr of trees to DT traverse with other (triangular number computation, see below)
constexpr unsigned nTrees = (1 << fieldTopLevel) * (1 << fieldTopLevel);
constexpr unsigned nComparisons = nTrees * (nTrees + 1) / 2;

namespace hdi::dr {
DualTree2dFieldComputation::DualTree2dFieldComputation() : _initialized(false), _w(0), _h(0) {
  // ...
}

DualTree2dFieldComputation::~DualTree2dFieldComputation() {
  if (_initialized) {
    clean();
  }
}

// Initialize gpu components for computation
void DualTree2dFieldComputation::initialize(const TsneParameters &params, unsigned n) {
  TIMERS_CREATE()
  _params = params;

  // Build shader programs
  try {
    for (auto &program : _programs) {
      program.create();
    }
    _programs[PROGRAM_DUAL_ROOT].addShader(COMPUTE, dt_root_src);
    _programs[PROGRAM_DUAL_TRAVERSE].addShader(COMPUTE, dt_tree_src);
    _programs[PROGRAM_DUAL_REDUCE].addShader(COMPUTE, dt_flatten_reduce_src);
    _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
    for (auto &program : _programs) {
      program.build();
    }
  } catch (const ShaderLoadingException &e) {
    std::cerr << e.what() << std::endl;
    exit(0); // yeah no, not recovering from this catastrophy
  }

  // Generate handles for buffers, textures etc
  glGenBuffers(_buffers.size(), _buffers.data());
  glGenTextures(_textures.size(), _textures.data());

  // Precompute unique pairs of left/right trees to compare
  // We do not compute these on the fly because of floating point
  // inaccuracies that occur with GLSL's sqrt() for very large indices.
  // For a fun read on triangular
  // numbers: https://en.wikipedia.org/wiki/Triangular_number
  {
    struct Comparison {
      std::uint32_t lidx;
      std::uint32_t ridx;
      std::uint32_t doSubdivide;
    };

    std::vector<Comparison> comparisons(nComparisons);

// nComparisons can become extremely large, so add trivial parallelism
#pragma omp parallel for
    for (int i = 0; i < nComparisons; i++) {
      // Compute "row" of triangle in which the n-most number appears
      // This is the floored inverse of (n (n + 1) / 2)
      std::uint32_t row =
          static_cast<std::uint32_t>(-0.5f + 0.5f * std::sqrt(1.0f + 8.0f * static_cast<float>(i)));

      // Compute first triangular number that appears in this row
      // This is again (n (n + 1) / 2)
      std::uint32_t base = row * (row + 1) / 2;

      // Compute indices of the trees we must compare
      comparisons[i] = {(nTrees - 1) - row, (nTrees - 1) - (i - base), 0u};
    }

    // Move data to immutable buffer storage
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_TREE_INDEX]);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(Comparison) * comparisons.size(),
                    comparisons.data(), 0);
  }

  // Set fields atlas texture array params
  glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_FIELD_ATLAS]);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Set fields texture params
  glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

  _quadtreeComputation.initialize(_params, n);

  _initialized = true;
  GL_ASSERT("DualTree2dFieldComputation::initialize");
}

// Remove gpu components
void DualTree2dFieldComputation::clean() {
  _quadtreeComputation.clean();
  TIMERS_DESTROY()
  glDeleteTextures(_textures.size(), _textures.data());
  glDeleteBuffers(_buffers.size(), _buffers.data());
  for (auto &program : _programs) {
    program.destroy();
  }
  _initialized = false;
  GL_ASSERT("DualTree2dFieldComputation::clean");
}

// Compute field
void DualTree2dFieldComputation::compute(unsigned w, unsigned h, float function_support,
                                         unsigned iteration, unsigned n, GLuint position_buff,
                                         GLuint bounds_buff, GLuint interp_buff, Bounds2D bounds) {
  Point2D minBounds = bounds.min;
  Point2D range = bounds.range();

  // Compute quad tree in atlas texture
  _quadtreeComputation.compute(w, h, n, position_buff, bounds_buff, bounds);

  // Rescale textures if necessary
  if (_w != w || _h != h) {
    _w = w;
    _h = h;

    auto atlasSize = _quadtreeComputation.getDensityAtlasSize();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_FIELD_ATLAS]);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, atlasSize.x, atlasSize.y, 3 * nArrayLevels, 0,
                 GL_RED, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_FIELD]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr);
  }

  // Clear out field atlas texture so we can add to it starting from 0
  glClearTexImage(_textures[TEXTURE_FIELD_ATLAS], 0, GL_RGBA, GL_FLOAT, nullptr);

  // Perform root subdivision for Barnes Hut approximation
  {
    TIMER_TICK(TIMER_DUAL_ROOT)
    auto &program = _programs[PROGRAM_DUAL_ROOT];
    program.bind();

    // Attach tree texture for sampling
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _quadtreeComputation.getDensityAtlasTexture());

    // Get atlas data
    Point2Dui atlasDims = _quadtreeComputation.getDensityAtlasSize();
    const unsigned atlasLevels = _quadtreeComputation.getDensityAtlasLevels().size();

    // Set uniforms
    program.uniform1ui("nComps", nComparisons);
    program.uniform1ui("nArrayLevels", nArrayLevelsRoot);
    program.uniform1ui("topLevel", atlasLevels - 1 - fieldTopLevel);
    program.uniform1f("theta2", theta * theta);
    program.uniform1i("treeAtlasSampler", 0);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _quadtreeComputation.getDensityAtlasBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_TREE_INDEX]);

    // Bind output uniform image
    glBindImageTexture(0, _textures[TEXTURE_FIELD_ATLAS], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

    // Perform computation
    glDispatchCompute(ceilDiv(nComparisons, 256u), 1u, 1u);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    program.release();
    TIMER_TOCK(TIMER_DUAL_ROOT)
    GL_ASSERT("DualTree2dFieldComputation::compute::dt_root")
  }

  // Compute field texture using dual tree Barnes Hut approximation
  {
    TIMER_TICK(TIMER_DUAL_TRAVERSE)
    auto &program = _programs[PROGRAM_DUAL_TRAVERSE];
    program.bind();

    // Attach tree texture for sampling
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _quadtreeComputation.getDensityAtlasTexture());

    // Get atlas data
    Point2Dui atlasDims = _quadtreeComputation.getDensityAtlasSize();
    const unsigned atlasLevels = _quadtreeComputation.getDensityAtlasLevels().size();

    // Set uniforms
    program.uniform1ui("nComps", nComparisons);
    program.uniform1ui("nArrayLevels", nArrayLevelsTravert);
    program.uniform1ui("topLevel", atlasLevels - 1 - fieldTopLevel);
    program.uniform1f("theta2", theta * theta);
    program.uniform1i("treeAtlasSampler", 0);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _quadtreeComputation.getDensityAtlasBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_TREE_INDEX]);

    // Bind output uniform image
    glBindImageTexture(0, _textures[TEXTURE_FIELD_ATLAS], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

    // Perform computation
    glDispatchCompute(ceilDiv(nComparisons, 32u), 1u, 1u); // 4x the nr of comparisons
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    program.release();
    TIMER_TOCK(TIMER_DUAL_TRAVERSE)
    GL_ASSERT("DualTree2dFieldComputation::compute::dt_traverse")
  }

  {
    TIMER_TICK(TIMER_REDUCE)
    auto &program = _programs[PROGRAM_DUAL_REDUCE];
    program.bind();

    // Bind buffers and write images
    glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _quadtreeComputation.getDensityAtlasBuffer());

    // Bind atlas texture to slot 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_FIELD_ATLAS]);

    // Get atlas data
    const unsigned atlasLevels = _quadtreeComputation.getDensityAtlasLevels().size();

    // Bind remaining uniforms
    program.uniform1ui("nLevels", atlasLevels - fieldTopLevel);
    program.uniform1ui("nArrayLevels", nArrayLevels);
    program.uniform2i("fieldTextureSize", _w, _h);
    program.uniform1i("fieldAtlasSampler", 0);

    // Perform computation
    glDispatchCompute(ceilDiv(_w, 16u), ceilDiv(_h, 16u), 1u);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);

    program.release();
    TIMER_TOCK(TIMER_REDUCE)
    GL_ASSERT("DualTree2dFieldComputation::compute::dt_flatten")
  }

  // Query field texture for positional values
  {
    TIMER_TICK(TIMER_INTERP)
    auto &program = _programs[PROGRAM_INTERP];
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
    TIMER_TOCK(TIMER_INTERP)
    GL_ASSERT("DualTree2dFieldComputation::compute::interp")
  }

  // TODO REMOVE DEBUG OUTPUT
  // Perform test texture output
  /* if (iteration % 100 == 0) {
    glActiveTexture(GL_TEXTURE0);
    const Point2Dui dims = _quadtreeComputation.getDensityAtlasSize();
    const auto& levels = _quadtreeComputation.getDensityAtlasLevels();

    // Output tree
    {
      const uint32_t nPixels = dims.x * dims.y;
      std::vector<float> _buffer(4 * nPixels);
      glBindTexture(GL_TEXTURE_2D, _quadtreeComputation.getDensityAtlasTexture());
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
        maxf = 1.f / maxf;
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

    // Obtain field atlas
    const uint32_t nPixels = dims.x * dims.y * nArrayLevels;
    std::vector<float> buffer(nPixels * 3, 0.f);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_FIELD_ATLAS]);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED, GL_FLOAT, buffer.data());

    // Output density atlas
    {
      std::vector<float> _buffer(nPixels);
      for (int i = 0; i < nArrayLevels; i++) {
        const auto it = buffer.begin() + (i * 3) * dims.x * dims.y;
        const auto _it = _buffer.begin() + i * dims.x * dims.y;
        std::copy(std::execution::par_unseq, it, it + dims.x * dims.y, _it);
      }
      auto maxf = 1.f / *std::max_element(_buffer.begin(), _buffer.end());
      std::transform(std::execution::par_unseq,
        _buffer.begin(), _buffer.end(), _buffer.begin(),
      [maxf] (const auto& f) {
        return f * maxf;
      });
      utils::valuesToImage("./images/density_" + std::to_string(iteration), _buffer, dims.x, dims.y
  * nArrayLevels, 1);
    }

    // Output combined channels gradient atlas
    {
      std::vector<float> _buffer(2 * nPixels, 0.f);
      for (int i = 0; i < 2; i++) {
        std::vector<float> __buffer(nPixels);
        for (int j = 0; j < nArrayLevels; j++) {
          const auto it = buffer.begin() + (j * 3 + i + 1) * dims.x * dims.y;
          const auto _it = __buffer.begin() + j * dims.x * dims.y;
          std::copy(std::execution::par_unseq, it, it + dims.x * dims.y, _it);
        }
        #pragma omp parallel for
        for (int j = 0; j < __buffer.size(); j++) {
          _buffer[2 * j + i] += __buffer[j];
        }
      }
      auto maxf = *std::max_element(_buffer.begin(), _buffer.end());
      auto minf = *std::min_element(_buffer.begin(), _buffer.end());
      auto rangef = 1.f / (maxf - minf);
      std::transform(std::execution::par_unseq, _buffer.begin(), _buffer.end(), _buffer.begin(),
      [minf, rangef] (const auto& f) {
        return 255.f * ((f - minf) * rangef);
      });
      utils::valuesToImage("./images/grad_" + std::to_string(iteration), _buffer, dims.x, dims.y *
  nArrayLevels, 2);
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
      utils::valuesToImage("./images/field_density_" + std::to_string(iteration), density, _w, _h,
  1); auto gMaxf = *std::max_element(gradient.begin(), gradient.end()); auto gMinf =
  *std::min_element(gradient.begin(), gradient.end()); auto gRangef = 1.f / (gMaxf - gMinf);
      std::transform(std::execution::par_unseq, gradient.begin(), gradient.end(), gradient.begin(),
      [gMaxf, gMinf, gRangef] (const auto& f) {
        return 255.f * ((f - gMinf) * gRangef);
      });
      utils::valuesToImage("./images/field_gradient_" + std::to_string(iteration), gradient, _w, _h,
  2);
    }
  } */

  // Update timers, log values on final iteration
  TIMERS_UPDATE()
  if (TIMERS_LOG_ENABLE && iteration >= _params._iterations - 1) {
    utils::secureLog(_logger, "\nField computation");
    TIMER_LOG(_logger, TIMER_DUAL_ROOT, "  Root")
    TIMER_LOG(_logger, TIMER_DUAL_TRAVERSE, "  Travert")
    TIMER_LOG(_logger, TIMER_REDUCE, "  Reduce")
    TIMER_LOG(_logger, TIMER_INTERP, "  Interp")
    utils::secureLog(_logger, "");
  }
}
} // namespace hdi::dr