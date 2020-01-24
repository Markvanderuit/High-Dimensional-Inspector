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
#include "2d_density_tree_computation.h"
#include "2d_density_tree_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/visual_utils.h"

// Magic numbers
constexpr unsigned maxNumLODLevels = 10;
constexpr float pointSize = 1.f;
constexpr unsigned numLayersPerIter = 4;

namespace hdi::dr {
  Density2dTreeComputation::Density2dTreeComputation() 
  : _logger(nullptr), _initialized(false), _iteration(0), _w(0), _h(0) {
    // ...
  }

  Density2dTreeComputation::~Density2dTreeComputation() {
    if (_initialized) {
      clean();
    }
  }

  void Density2dTreeComputation::initialize(const TsneParameters& params,
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
      int i = -1;
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

  void Density2dTreeComputation::clean() {
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

  void Density2dTreeComputation::compute(unsigned w, unsigned h, unsigned n,
                GLuint position_buff, GLuint bounds_buff,
                Bounds2D bounds) {   
    // Rescale textures if necessary
    if (_w != w || _h != h) {
      _w = w;
      _h = h;
      
      // Compute mipmap LOD dimensions starting from 0, current level
      std::vector<Point2Dui> dimensions;
      for (int i = 0; i < maxNumLODLevels; i++) {
        unsigned wLod = std::max(static_cast<unsigned>(_w / std::pow(2, i)), 1u);
        unsigned hLod = std::max(static_cast<unsigned>(_h / std::pow(2, i)), 1u);
        dimensions.push_back(Point2Dui { wLod, hLod });
        if (wLod == 1u && hLod == 1u) {
          break;
        }
      }

      // Generate atlas dimensions
      _atlasLevels.resize(dimensions.size());
      _atlasLevels[0] = {  Point2Dui { 0, 0 }, dimensions[0] };
      Point2Dui offset { 0, dimensions[0].y };
      for (int i = 1; i < _atlasLevels.size(); i++) {
        _atlasLevels[i] = { offset,  dimensions[i] };
        offset.x += dimensions[i].x;
      }
      _atlasDimensions = {
        _atlasLevels[0].size.x,
        _atlasLevels[0].size.y + _atlasLevels[1].size.y
      };

      // Move atlas dimensions buffer to device
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_ATLAS_DATA]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, _atlasLevels.size() * sizeof(AtlasLevelBlock), _atlasLevels.data(), GL_STATIC_READ);

      // Generate atlas texture with encompassing dimensions
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
      GL_ASSERT("Density2dTreeComputation::compute::DENSITY_LOD_BOTTOM");
    }

    // Divide centers of mass by n for every pixel
    {
      TIMER_TICK(TIMER_DENSITY_LOD_BOTTOM_DIVIDE)
      auto& program = _programs[PROGRAM_DENSITY_LOD_BOTTOM_DIVIDE];
      program.bind();

      // Bind atlas to texture slot 0 for nice cached reading
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      program.uniform1i("texture32f", 0);

      // Bind texture as image for writing
      glBindImageTexture(0, _textures[TEXTURE_DENSITY_ATLAS_32F], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      
      // Bind buffers to shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_ATLAS_DATA]);

      // Perform computation
      // 8x8 amounts to 64 units per work group
      const auto& dims = _atlasLevels[0].size;
      glDispatchCompute((dims.x / 8) + 1, (dims.y / 8) + 1, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

      program.release();
      TIMER_TOCK(TIMER_DENSITY_LOD_BOTTOM_DIVIDE)
      GL_ASSERT("Density2dTreeComputation::compute::DENSITY_LOD_BOTTOM_DIVIDE");
    }

    // Generate upper LOD level density
    {
      TIMER_TICK(TIMER_DENSITY_LOD_UPPER)
      auto& program = _programs[PROGRAM_DENSITY_LOD_UPPER];
      program.bind();

      // Bind atlas to texture slot 0 for nice cached reading
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      program.uniform1i("texture32f", 0);

      // Also bind as image for writing
      glBindImageTexture(0, _textures[TEXTURE_DENSITY_ATLAS_32F], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      
      // Bind buffer to shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_ATLAS_DATA]);

      // Compute LOD levels iteratively, synchronizing in between
      for (int l = 1; l < _atlasLevels.size(); l += numLayersPerIter) {
        // Set uniforms for the range of layers we want to compute
        program.uniform1i("l", l);
        program.uniform1i("maxl", std::min<int>(l + numLayersPerIter + 1, _atlasLevels.size()));

        // 8x8 amounts to 64 units per work group
        const auto& dims = _atlasLevels[l].size;
        glDispatchCompute((dims.x / 8u) + 1, (dims.y / 8u) + 1, 1u);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      }

      program.release();
      TIMER_TOCK(TIMER_DENSITY_LOD_UPPER)
      GL_ASSERT("Density2dTreeComputation::compute::PROGRAM_DENSITY_LOD_UPPER");
    }

    /* { // Debug output for density atlas
      std::vector<float> buffer32f(_atlasDimensions.x * _atlasDimensions.y * 4, 0.f);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer32f.data());

      {
        const auto& dim = _atlasLevels[_atlasLevels.size() - 1].offset;
        const auto idx = dim.y * (4 * _atlasDimensions.x) + dim.x * 4;
        const auto i = static_cast<unsigned>(buffer32f[idx]);
        if (i != n) {
          std::stringstream msg;
          msg << "Output density at top level was "
              << i 
              << ", but should be "
              << n
              << ", during iteration "\
              << _iteration
              << ", with resolution "
              << _atlasDimensions.x << " " << _atlasDimensions.y;
          std::cerr << msg.str() << std::endl;

          // Output iteration
          std::vector<float> image32f(_atlasDimensions.x * _atlasDimensions.y, 0.f);
          for (int y = 0; y < _atlasDimensions.y; y++) {
            for (int x = 0; x < _atlasDimensions.x; x++) {
                const auto idx = y * (4 * _atlasDimensions.x) + x * 4;
                float f = buffer32f[idx];
                image32f[y * _atlasDimensions.x + x] = (f > 0.f) ? 255.f : 0.f;
            }
          }
          utils::valuesToImage("./images/densityAtlas_" + std::to_string(_iteration), image32f, _atlasDimensions.x, _atlasDimensions.y, 1);
      
          // throw std::runtime_error(msg.str());
        } else {
          // std::stringstream msg;
          // msg << "All fine with resolution "
          //     << _atlasDimensions.x << " " << _atlasDimensions.y;
          // std::cerr << msg.str() << std::endl;
        }
      }
    } */

    // TODO Remove debugging output
    // Output density texture values for debugging
    /* if (_iteration == 999) {
      // Copy texture array
      std::vector<float> buffer32f(_atlasDimensions.x * _atlasDimensions.y * 4, 0.f);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DENSITY_ATLAS_32F]);
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer32f.data());
      // for (int y = 0; y < _atlasDimensions.y; y++) {
      //   for (int x = 0; x < _atlasDimensions.x; x++) {
      //       const auto idx = y * (4 * _atlasDimensions.x) + x * 4;
      //       const auto i = static_cast<unsigned>(buffer32f[idx]);
      //       const auto fx = static_cast<float>(buffer32f[idx + 1]);
      //       const auto fy = static_cast<float>(buffer32f[idx + 1]);
      //       // printf("%6i", i);
      //       printf("(%1.3f,%1.3f)", fx, fy);
      //   }
      //   printf("\n");
      // }
      {
        const auto& dim = _atlasLevels[_atlasLevels.size() - 1].offset;
        const auto idx = dim.y * (4 * _atlasDimensions.x) + dim.x * 4;
        const auto i = static_cast<unsigned>(buffer32f[idx]);
        if (i != n) {
          std::stringstream msg;
          msg << "Output density at top level was "
              << i 
              << ", should be "
              << n;
          throw std::runtime_error(msg.str());
        }
      }
      
      // Store image
      std::vector<float> image32f(_atlasDimensions.x * _atlasDimensions.y, 0.f);
      for (int y = 0; y < _atlasDimensions.y; y++) {
        for (int x = 0; x < _atlasDimensions.x; x++) {
            const auto idx = y * (4 * _atlasDimensions.x) + x * 4;
            float f = buffer32f[idx];
            image32f[y * _atlasDimensions.x + x] = (f > 0.f) ? 255.f : 0.f;
        }
      }
      utils::valuesToImage("./images/densityAtlas", image32f, _atlasDimensions.x, _atlasDimensions.y, 1);
      image32f.resize(_atlasDimensions.x * _atlasDimensions.y * 2);
      for (int y = 0; y < _atlasDimensions.y; y++) {
        for (int x = 0; x < _atlasDimensions.x; x++) {
            const auto idx = y * (4 * _atlasDimensions.x) + x * 4;
            float fx = buffer32f[idx + 1];
            float fy = buffer32f[idx + 2];
            image32f[y * 2 * _atlasDimensions.x + x * 2] = fx * 255.f;
            image32f[y * 2 * _atlasDimensions.x + x * 2 + 1] = fx * 255.f;
        }
      }
      utils::valuesToImage("./images/densityCOM", image32f, _atlasDimensions.x, _atlasDimensions.y, 2);

      // Output LOD dimensions
      std::stringstream ss;
      ss << "Atlas dimensions: ";
      for (const auto& level : _atlasLevels) {
        ss << "{(" << level.offset.x << "," << level.offset.y << "), "
           << "(" << level.size.x << "," << level.size.y << ")} ";
      } 
      std::cout << ss.str() << std::endl;
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