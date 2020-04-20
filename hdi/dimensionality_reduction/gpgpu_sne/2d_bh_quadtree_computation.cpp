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
#include "2d_bh_quadtree_computation.h"
#include "2d_bh_quadtree_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/visual_utils.h"

// Magic numbers
constexpr float pointSize = 1.f;
constexpr unsigned minNumLODLevels = 7; // lowest level is 64^2
constexpr unsigned maxNumLODLevels = 12; // lowest level is 2048^2
constexpr unsigned numLayersPerIter = 5;

namespace hdi::dr {
  BarnesHut2dQuadtreeComputation::BarnesHut2dQuadtreeComputation() 
  : _logger(nullptr), _initialized(false), _iteration(0), _w(0), _h(0), _baseLevel(0) {
    // ...
  }

  BarnesHut2dQuadtreeComputation::~BarnesHut2dQuadtreeComputation() {
    if (_initialized) {
      clean();
    }
  }

  void BarnesHut2dQuadtreeComputation::initialize(const TsneParameters& params,
                                            unsigned n) {
    TIMERS_CREATE()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_DENSITY_LOD_BOTTOM].addShader(VERTEX, density_lod_bottom_vertex);
      _programs[PROGRAM_DENSITY_LOD_BOTTOM].addShader(FRAGMENT, density_lod_bottom_fragment);
      _programs[PROGRAM_DENSITY_LOD_BOTTOM_DIVIDE].addShader(COMPUTE, density_lod_bottom_divide_comp);
      _programs[PROGRAM_DENSITY_LOD_UPPER].addShader(COMPUTE, density_lod_upper_src);
      // Add shaders
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // not rceovering from this, and exceptions don't reach top-level
    }
    
    // Configure textures
    glGenTextures(_textures.size(), _textures.data());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    // Configure buffers
    // Atlas gets created in compute() when resolution changes
    glGenBuffers(_buffers.size(), _buffers.data());

    // Configure framebuffer
    glGenFramebuffers(_framebuffers.size(), _framebuffers.data());
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_DENSITY_LOD_BOTTOM]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[TEXTURE_DENSITY_ATLAS_32F], 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;
    GL_ASSERT("Density2dTreeComputation::initialize");
  }

  void BarnesHut2dQuadtreeComputation::clean() {
    TIMERS_DESTROY()
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(_framebuffers.size(), _framebuffers.data());
    glDeleteVertexArrays(1, &_point_vao);
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
    GL_ASSERT("Density2dTreeComputation::clean");
  }

  void BarnesHut2dQuadtreeComputation::compute(unsigned w, unsigned h, unsigned n,
                                         GLuint position_buff, GLuint bounds_buff,
                                         Bounds2D bounds) {
    // The tree has a minimum depth, compute minimum width given this depth
    const unsigned topLODMinSize = (std::pow(2u, minNumLODLevels - 1));
    const unsigned wCapped = std::max(w, topLODMinSize);
    const unsigned hCapped = std::max(h, topLODMinSize);

    // The base level of the 'normal' resolution is then not necessarily 0
    _baseLevel = static_cast<unsigned>(std::log2(wCapped / w));

    // Rescale textures only if necessary
    if (_w != wCapped || _h != hCapped) {
      _w = wCapped;
      _h = hCapped;

      // Compute texture atlas dimensions for different levels
      std::vector<unsigned> dimensions;
      const unsigned iMax = std::min(_w, 
        static_cast<unsigned>(std::pow(2, maxNumLODLevels - 1u)));
      for (unsigned i = 1u; i <= iMax; i *= 2u) {
        dimensions.push_back(i);
      }
      std::reverse(dimensions.begin(), dimensions.end());

      // Compute texture atlas offsets for different levels
      _atlasLevels.resize(dimensions.size());
      _atlasLevels[0] = { { 0, 0 }, { dimensions[0], dimensions[0] } };
      Point2Dui offset { 0, dimensions[0] };
      for (int i = 1; i < _atlasLevels.size(); i++) {
        _atlasLevels[i] = { offset, { dimensions[i], dimensions[i] } };
        offset.x += dimensions[i];
      }
      _atlasDimensions = {
        _atlasLevels[0].size.x,
        _atlasLevels[0].size.y + _atlasLevels[1].size.y
      };

      // Copy atlas dimensions buffer to device
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_ATLAS_DATA]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, _atlasLevels.size() * sizeof(AtlasLevelBlock), _atlasLevels.data(), GL_STATIC_READ);

      // Generate texture exactly fitting all atlas levels
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _atlasDimensions.x, _atlasDimensions.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Perform initial density computation by splatting points
    {
      TIMER_TICK(TIMER_DENSITY_LOD_BOTTOM)
      auto& program = _programs[PROGRAM_DENSITY_LOD_BOTTOM];
      program.bind();
      
      // Bind buffers to shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);

      // Prepare for drawing
      const auto& dims = _atlasLevels[0].size;
      glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_DENSITY_LOD_BOTTOM]);
      glEnable(GL_BLEND);
      glBlendFunc(GL_ONE, GL_ONE);
      glViewport(0, 0, dims.x, dims.y);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glPointSize(pointSize);
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);

      // Draw embedding points
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_BLEND);

      program.release();
      TIMER_TOCK(TIMER_DENSITY_LOD_BOTTOM)
      GL_ASSERT("Density2dTreeComputation::compute::density_lod_bottom");
    }

    // Divide centers of mass by n for every pixel
    {
      TIMER_TICK(TIMER_DENSITY_LOD_BOTTOM_DIVIDE)
      auto& program = _programs[PROGRAM_DENSITY_LOD_BOTTOM_DIVIDE];
      program.bind();
      
      // Bind texture as image for read/write
      glBindImageTexture(0, _textures[TEXTURE_DENSITY_ATLAS_32F], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
      
      // Bind buffers to shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_ATLAS_DATA]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);

      // Perform computation
      const auto& dims = _atlasLevels[0].size;
      glDispatchCompute(ceilDiv(dims.x, 16u), ceilDiv(dims.y, 16u), 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

      program.release();
      TIMER_TOCK(TIMER_DENSITY_LOD_BOTTOM_DIVIDE)
      GL_ASSERT("Density2dTreeComputation::compute::density_lod_bottom_divide");
    }

    // Generate upper LOD levels through reduction
    {
      TIMER_TICK(TIMER_DENSITY_LOD_UPPER)
      auto& program = _programs[PROGRAM_DENSITY_LOD_UPPER];
      program.bind();

      // Bind atlas to texture slot 0 for nice cached reading
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      program.uniform1i("texture32f", 0);
      program.uniform1ui("nLevels", _atlasLevels.size());

      // Also bind as image for writing
      glBindImageTexture(0, _textures[TEXTURE_DENSITY_ATLAS_32F], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      
      // Bind buffer to shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_ATLAS_DATA]);

      // Compute LOD levels iteratively, synchronizing in between reductions
      for (int l = 1; l < _atlasLevels.size(); l += numLayersPerIter) {
        // Set uniforms for the nr of levels we want to reduce
        program.uniform1ui("l", l);
        program.uniform1ui("maxl", std::min<unsigned>(l + numLayersPerIter + 1, _atlasLevels.size()));

        // Perform computation
        const auto& dims = _atlasLevels[l].size;
        glDispatchCompute(ceilDiv(dims.x, 16u), ceilDiv(dims.y, 16u), 1u);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      }

      program.release();
      TIMER_TOCK(TIMER_DENSITY_LOD_UPPER)
      GL_ASSERT("Density2dTreeComputation::compute::density_lod_upper");
    }

    /* if (_iteration % 100 == 0) {
      const auto dims = _atlasDimensions;
      std::vector<float> buffer(4 * dims.x * dims.y);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer.data());
      const auto px = _atlasLevels[_atlasLevels.size() - 1].offset;
      const unsigned idx = px.y * dims.x + px.x;
      std::cout << px.x << " " << px.y << std::endl;
      std::cout << buffer[4 * idx] << std::endl;
    } */

    // Update timers, log values on final iteration
    TIMERS_UPDATE()
    if (TIMERS_LOG_ENABLE && _iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "\nTree building");
      TIMER_LOG(_logger, TIMER_DENSITY_LOD_BOTTOM, "  Bottom")
      TIMER_LOG(_logger, TIMER_DENSITY_LOD_BOTTOM_DIVIDE, "  Divide")
      TIMER_LOG(_logger, TIMER_DENSITY_LOD_UPPER, "  Upper")
    }
    _iteration++;
    GL_ASSERT("Density2dTreeComputation::compute");
  }
}