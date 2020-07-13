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
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_3d.h"

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

// Magic numbers
constexpr float pointSize = 2.f;

namespace hdi::dr {
  Field3dCompute::Field3dCompute()
  : _isInit(false), _dims(0), _logger(nullptr) {
#ifdef USE_BVH
    _rebuildBvhOnIter = true;
    _nRebuildIters = 0;
    _lastRebuildTime = 0.0;
#endif
  }

  Field3dCompute::~Field3dCompute() {
    if (_isInit) {
      destr();
    }
  }

  void Field3dCompute::init(const TsneParameters& params, 
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
      _programs(ProgramType::eGrid).addShader(VERTEX, grid_vert_src);
      _programs(ProgramType::eGrid).addShader(FRAGMENT, grid_fragment_src);
#ifdef USE_BVH
      _programs(ProgramType::eField).addShader(COMPUTE, field_bvh_src); 
#else
      _programs(ProgramType::eField).addShader(COMPUTE, field_src); 
#endif
      _programs(ProgramType::eInterp).addShader(COMPUTE, interp_src);
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

    // Generate voxel grid cell texture
    glCreateTextures(GL_TEXTURE_1D, 1, &_textures(TextureType::eCellmap));
    glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    // Generate voxel grid texture
    glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eGrid));
    glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 3d fields texture
    glCreateTextures(GL_TEXTURE_3D, 1, &_textures(TextureType::eField));
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Generate framebuffer for grid computation
    glCreateFramebuffers(1, &_frbo_grid);
    glNamedFramebufferTexture(_frbo_grid, GL_COLOR_ATTACHMENT0, _textures(TextureType::eGrid), 0);
    glNamedFramebufferDrawBuffer(_frbo_grid, GL_COLOR_ATTACHMENT0);
    
    // Generate vrao for point drawing
    glCreateVertexArrays(1, &_vrao_point);
    glVertexArrayVertexBuffer(_vrao_point, 0, position_buff, 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_vrao_point, 0);
    glVertexArrayAttribFormat(_vrao_point, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vrao_point, 0, 0);

#ifdef USE_BVH
    _bvh.init(params, BVH<3>::Layout(n, 8, 4), position_buff, bounds_buff);
    _renderer.init(_bvh, bounds_buff);
#endif

    _isInit = true;
  }

  void Field3dCompute::destr() {
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
    _isInit = false;
  }

  void Field3dCompute::compute(uvec dims, float function_support, unsigned iteration, unsigned n,
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
        glBindTexture(GL_TEXTURE_1D, _textures(TextureType::eCellmap));
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI, _dims.z, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, _cellData.data());
      }
      glBindTexture(GL_TEXTURE_2D, _textures(TextureType::eGrid));
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, _dims.x, _dims.y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
      glBindTexture(GL_TEXTURE_3D, _textures(TextureType::eField));
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _dims.x, _dims.y, _dims.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Compute grid texture
    {
      TICK_TIMER(TIMR_GRID)
      
      auto& program = _programs(ProgramType::eGrid);
      program.bind();
      program.uniform1i("cellMap", 0);
      program.uniform1f("zPadding", 0.025f);

      // Bind textures and buffers
      glBindTextureUnit(0, _textures(TextureType::eCellmap));
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

#ifdef USE_BVH
    _bvh.compute(_rebuildBvhOnIter, iteration, position_buff, bounds_buff);
#endif

    // Compute fields 3D texture through O(px n) reduction
    {
      TICK_TIMER(TIMR_FIELD)

      auto &program = _programs(ProgramType::eField);
      program.bind();

#ifdef USE_BVH
      // Query bvh for layout and buffer object handles
      const auto layout = _bvh.layout();
      const auto buffers = _bvh.buffers();

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bounds_buff);

      // Bind textures and images
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      glBindTextureUnit(0, _textures(TextureType::eGrid));

      // Set uniforms
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("nLvls", layout.nLvls);
      program.uniform1ui("kNode", layout.nodeFanout);
      program.uniform1ui("kLeaf", layout.leafFanout);
      program.uniform1ui("gridDepth", std::min(128u, _dims.z));
      program.uniform1f("theta", .5f); // TODO extract and make program parameter
      program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(_dims.x, 4u), ceilDiv(_dims.y, 4u), ceilDiv(_dims.z, 4u));
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

#else
      // Bind textures, images and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindTextureUnit(0, _textures(TextureType::eGrid));
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

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

      auto &program = _programs(ProgramType::eInterp);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform3ui("fieldSize", _dims.x, _dims.y, _dims.z);
      program.uniform1i("fieldSampler", 0);

      // Bind textures and buffers for this shader
      glBindTextureUnit(0, _textures(TextureType::eField));
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
    if (iteration >= _params._iterations - 1) {
      utils::secureLog(_logger, "\nField computation");
      LOG_TIMER(_logger, TIMR_GRID, "  Grid")
      LOG_TIMER(_logger, TIMR_FIELD, "  Field")
      LOG_TIMER(_logger, TIMR_INTERP, "  Interp")
    }
    
#ifdef USE_BVH
    // _rebuildBvhOnIter = true;
    // Rebuild BVH once field computation time diverges from the last one by a certain degree
    /* const uint maxIters = 6;
    const double threshold = 0.01;// 0.0075; // how the @#@$@ do I determine this well
    if (_rebuildBvhOnIter && _iteration >= _params._remove_exaggeration_iter) {
      _lastRebuildTime = glTimers[TIMR_FIELD].lastMicros();
      _nRebuildIters = 0;
      _rebuildBvhOnIter = false;
    } else if (_lastRebuildTime > 0.0) {
      const double difference = std::abs(glTimers[TIMR_FIELD].lastMicros() - _lastRebuildTime) 
                              / _lastRebuildTime;
      if (difference >= threshold || ++_nRebuildIters >= maxIters) {
        _rebuildBvhOnIter = true;
      }
    } */
    _rebuildBvhOnIter = true;
#endif
  }
}