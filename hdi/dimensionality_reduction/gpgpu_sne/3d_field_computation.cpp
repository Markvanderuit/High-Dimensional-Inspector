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
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_field_shaders.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_field_computation.h"

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

constexpr float pointSize = 2.f;

namespace hdi::dr {
  Baseline3dFieldComputation::Baseline3dFieldComputation()
  : _isInit(false), _iteration(0), _dims(0), _logger(nullptr) {
#ifdef USE_BVH
    _rebuildBvhOnIter = true;
    _lastRebuildBvhTime = 0.0;
#endif
  }

  Baseline3dFieldComputation::~Baseline3dFieldComputation() {
    if (_isInit) {
      destr();
    }
  }

  void Baseline3dFieldComputation::init(const TsneParameters& params, 
                                              GLuint position_buff,
                                              GLuint bounds_buff, 
                                              unsigned n) {
    INIT_TIMERS()
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROG_GRID].addShader(VERTEX, grid_vert_src);
      _programs[PROG_GRID].addShader(FRAGMENT, grid_fragment_src);
#ifdef USE_BVH
      _programs[PROG_FIELD].addShader(COMPUTE, field_bvh_src); 
#else
      _programs[PROG_FIELD].addShader(COMPUTE, field_src); 
#endif
      _programs[PROG_INTERP].addShader(COMPUTE, interp_src);
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
    glCreateFramebuffers(1, &_frbo_grid);
    glNamedFramebufferTexture(_frbo_grid, GL_COLOR_ATTACHMENT0, _textures[TEXTURE_GRID], 0);
    glNamedFramebufferDrawBuffer(_frbo_grid, GL_COLOR_ATTACHMENT0);
    
    // Generate vrao for point drawing
    glCreateVertexArrays(1, &_vrao_point);
    glVertexArrayVertexBuffer(_vrao_point, 0, position_buff, 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_vrao_point, 0);
    glVertexArrayAttribFormat(_vrao_point, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vrao_point, 0, 0);

#ifdef USE_BVH
    _bvh.init(params, position_buff, bounds_buff, n);
    _renderer.init(_bvh, bounds_buff);
#endif

    _iteration = 0;
    _isInit = true;
  }

  void Baseline3dFieldComputation::destr() {
    DSTR_TIMERS()
#ifdef USE_BVH
    _renderer.destr();
    _bvh.destr();
#endif
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_frbo_grid);
    glDeleteVertexArrays(1, &_vrao_point);
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _isInit = false;
  }

  void Baseline3dFieldComputation::compute(uvec dims, float function_support, unsigned n,
                                           GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                                           Bounds bounds) {
    vec minBounds = bounds.min;
    vec range = bounds.range();

    // Rescale textures if necessary
    if (_dims != dims) {
      _dims = dims;

      // Update cell data for voxel grid computation only to d = 128
      glActiveTexture(GL_TEXTURE0);
      if (_dims.z <= 128u) {
        glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI, _dims.z, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, _cellData.data());
      }
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, _dims.x, _dims.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD]);
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _dims.x, _dims.y, _dims.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

#ifdef USE_BVH
    // Compute BVH structure over embedding
    _bvh.compute(_rebuildBvhOnIter, _iteration);
#endif

    // Compute grid texture
    {
      TICK_TIMER(TIMR_GRID)
      
      auto& program = _programs[PROG_GRID];
      program.bind();
      program.uniform1i("cellMap", 0);
      program.uniform1f("zPadding", 0.025f);

      // Bind textures and buffers
      glBindTextureUnit(0, _textures[TEXTURE_CELLMAP]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Specify and clear framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, _frbo_grid);
      constexpr std::array<float, 4> clearColor = { 0.f, 0.f, 0.f, 0.f };
      constexpr float clearDepth = 1.f;
      glClearNamedFramebufferfv(_frbo_grid, GL_COLOR, 0, clearColor.data());
      glClearNamedFramebufferfv(_frbo_grid, GL_DEPTH, 0, &clearDepth);

      // Specify logic test, viewport and point size
      glViewport(0, 0, _dims.x, _dims.y);
      glEnable(GL_COLOR_LOGIC_OP);
      glLogicOp(GL_OR);
      glPointSize(pointSize);
      
      // Draw a point at each embedding position
      glBindVertexArray(_vrao_point);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glDisable(GL_COLOR_LOGIC_OP);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      
      TOCK_TIMER(TIMR_GRID)
    }

    // Compute fields 3D texture through O(px n) reduction
    {
      TICK_TIMER(TIMR_FIELD)

      auto &program = _programs[PROG_FIELD];
      program.bind();

#ifdef USE_BVH
      // Query bvh for layout specs and memory structure
      const auto& layout = _bvh.layout();
      const auto& memr = _bvh.memr();

      // Bind buffers
      memr.bindBuffer(bvh::BVHExtMemr<3>::MemrType::eNode, 0);
      memr.bindBuffer(bvh::BVHExtMemr<3>::MemrType::eIdx, 1);
      memr.bindBuffer(bvh::BVHExtMemr<3>::MemrType::ePos, 2);
      memr.bindBuffer(bvh::BVHExtMemr<3>::MemrType::eMinB, 3);
      memr.bindBuffer(bvh::BVHExtMemr<3>::MemrType::eDiam, 4);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bounds_buff);

      // Bind textures and images
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      glBindTextureUnit(0, _textures[TEXTURE_GRID]);

      // Set uniforms
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("kNode", layout.kNode);
      program.uniform1ui("nLvls", layout.nLvls);
      program.uniform1ui("gridDepth", std::min(128u, _dims.z));
      program.uniform1f("theta", 1.f);
      program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(_dims.x, 4u), ceilDiv(_dims.y, 4u), ceilDiv(_dims.z, 4u));
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

#else
      // Bind textures, images and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindTextureUnit(0, _textures[TEXTURE_GRID]);
      glBindImageTexture(0, _textures[TEXTURE_FIELD], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Set uniforms
      program.uniform1ui("num_points", n);
      program.uniform1ui("grid_depth", std::min(128u, _dims.z));
      program.uniform3ui("texture_size", _dims.x, _dims.y, _dims.z);
      program.uniform1i("grid_texture", 0);

      // Dispatch compute shader
      glDispatchCompute(_dims.x, _dims.y, 2);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
#endif
      
      TOCK_TIMER(TIMR_FIELD)
    }

    // Query field texture for positional values
    {
      TICK_TIMER(TIMR_INTERP)

      auto &program = _programs[PROG_INTERP];
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform3ui("fieldSize", _dims.x, _dims.y, _dims.z);
      program.uniform1i("fieldSampler", 0);

      // Bind textures and buffers for this shader
      glBindTextureUnit(0, _textures[TEXTURE_FIELD]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(n, 128u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      TOCK_TIMER(TIMR_INTERP)
    }
    
    // Update timers, log values on final iteration
    POLL_TIMERS()
    if (_iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "\nField computation");
      LOG_TIMER(_logger, TIMR_GRID, "  Grid")
      LOG_TIMER(_logger, TIMR_FIELD, "  Field")
      LOG_TIMER(_logger, TIMR_INTERP, "  Interp")
    }
// #ifdef USE_BVH
    // Rebuild BVH once field computation time diverges from the last one by a certain degree
    // if (_rebuildBvhOnIter) {
    //   _lastRebuildBvhTime = glTimers[TIMR_FIELD].lastMicros();
    //   if (_iteration >= _params._remove_exaggeration_iter) {
    //     _rebuildBvhOnIter = false;
    //     // std::cout << _iteration << std::endl;
    //   }
    // } else if (_lastRebuildBvhTime > 0.0) {
    //   double currentRebuildTime = glTimers[TIMR_FIELD].lastMicros();
    //   if (std::abs(currentRebuildTime - _lastRebuildBvhTime) / _lastRebuildBvhTime > 0.01) {
    //     _rebuildBvhOnIter = true;
    //   }
    // }
// #endif

    _iteration++;
  }
}