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
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_2d.h"

#define USE_WIDE_BVH // Toggle full-warp traversal of a wide BVH

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

// Magic numbers
constexpr float pointSize = 3.f;

namespace hdi::dr {
  Field2dCompute::Field2dCompute()
  : _isInit(false), 
    _dims(0), 
    _logger(nullptr),
    _useBvh(false),
    _rebuildBvhOnIter(true),
    _nRebuildIters(0),
    _lastRebuildTime(0.0) { }

  Field2dCompute::~Field2dCompute() {
    if (_isInit) {
      destr();
    }
  }

  void Field2dCompute::init(const TsneParameters& params, 
                            GLuint position_buff,
                            GLuint bounds_buff, 
                            unsigned n) {
    _params = params;
    _useBvh = _params._theta > 0.0;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }

      _programs(ProgramType::eStencil).addShader(VERTEX, stencil_vert_src);
      _programs(ProgramType::eStencil).addShader(FRAGMENT, stencil_fragment_src);
      _programs(ProgramType::eInterp).addShader(COMPUTE, interp_src);

      if (_useBvh) {
#ifdef USE_WIDE_BVH
        _programs(ProgramType::eField).addShader(COMPUTE, field_bvh_wide_src); 
#else
        _programs(ProgramType::eField).addShader(COMPUTE, field_bvh_src); 
#endif // USE_WIDE_BVH
      } else {
        _programs(ProgramType::eField).addShader(COMPUTE, field_src); 
      }
      
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }
    
    // Generate textures
    glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Generate framebuffer for grid computation
    glCreateFramebuffers(1, &_frbo_stencil);
    glNamedFramebufferTexture(_frbo_stencil, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
    glNamedFramebufferDrawBuffer(_frbo_stencil, GL_COLOR_ATTACHMENT0);

    // Generate vrao for point drawing
    glCreateVertexArrays(1, &_vrao_point);
    glVertexArrayVertexBuffer(_vrao_point, 0, position_buff, 0, 2 * sizeof(float));
    glEnableVertexArrayAttrib(_vrao_point, 0);
    glVertexArrayAttribFormat(_vrao_point, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vrao_point, 0, 0);

    if (_useBvh) {
#ifdef USE_WIDE_BVH
    _bvh.init(params, BVH<2>::Layout(n, 16, 4), position_buff, bounds_buff);
#else
    _bvh.init(params, BVH<2>::Layout(n, 4, 4), position_buff, bounds_buff);
#endif // USE_WIDE_BVH
    // _renderer.init(_bvh, bounds_buff);
    }

    INIT_TIMERS()
    ASSERT_GL("Field2dCompute::init()");
    _isInit = true;
  }

  // Remove gpu components
  void Field2dCompute::destr() {
    if (_useBvh) {
      _bvh.destr();
    // _renderer.destr();
    }
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_frbo_stencil);
    glDeleteVertexArrays(1, &_vrao_point);
    for (auto& program : _programs) {
      program.destroy();
    }
    DSTR_TIMERS();
    ASSERT_GL("Field2dCompute::destr()");
    _isInit = false;
  }

  // Compute field and depth textures
  void Field2dCompute::compute(uvec dims, float function_support, unsigned iteration, unsigned n,
                                           GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                                           Bounds bounds) {
    vec minBounds = bounds.min;
    vec range = bounds.range();

    // Rescale textures if necessary
    if (_dims != dims) {
      _dims = dims;

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures(TextureType::eStencil));
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, _dims.x, _dims.y, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
      glBindTexture(GL_TEXTURE_2D, _textures(TextureType::eField));
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _dims.x, _dims.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Compute BVH structure over embedding
    if (_useBvh) {
      _bvh.compute(_rebuildBvhOnIter, iteration, position_buff, bounds_buff);
    }

    // Compute stencil texture
    {
      TICK_TIMER(TIMR_STENCIL)

      auto& program = _programs(ProgramType::eStencil);
      program.bind();

      // Specify and clear framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, _frbo_stencil);
      constexpr std::array<float, 4> clearColor = { 0.f, 0.f, 0.f, 0.f };
      constexpr float clearDepth = 1.f;
      glClearNamedFramebufferfv(_frbo_stencil, GL_COLOR, 0, clearColor.data());
      glClearNamedFramebufferfv(_frbo_stencil, GL_DEPTH, 0, &clearDepth);

      // Specify viewport and point size, bind bounds buffer
      glViewport(0, 0, _dims.x, _dims.y);
      glPointSize(pointSize);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Draw a point at each embedding position
      glBindVertexArray(_vrao_point);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      
      ASSERT_GL("Field2dCompute::compute::stencil()");
      TOCK_TIMER(TIMR_STENCIL)
    }

    // Compute fields texture
    {
      TICK_TIMER(TIMR_FIELD)

      auto& program = _programs(ProgramType::eField);
      program.bind();

      glBindTextureUnit(0, _textures(TextureType::eStencil));
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      if (_useBvh) {
        // Grab layout specs and buffer object handles
        const auto layout = _bvh.layout();
        const auto buffers = _bvh.buffers();

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bounds_buff);

        // Set uniforms
        program.uniform1ui("nPos", layout.nPos);
        program.uniform1ui("nLvls", layout.nLvls);
        program.uniform1ui("kNode", layout.nodeFanout);
        program.uniform1ui("kLeaf", layout.leafFanout);
        program.uniform1f("theta2", _params._theta * _params._theta); // TODO extract and make program parameter
        program.uniform2ui("textureSize", _dims.x, _dims.y);
        
      // Dispatch compute shader
#ifdef USE_WIDE_BVH
      glDispatchCompute(ceilDiv(_dims.x, 32u) * layout.nodeFanout, ceilDiv(_dims.y, 4u), 1);
#else
      glDispatchCompute(ceilDiv(_dims.x, 16u), ceilDiv(_dims.y, 16u), 1);
#endif // USE_WIDE_BVH
      } else {
        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
        
        // Set uniforms
        program.uniform1ui("num_points", n);
        program.uniform2ui("texture_size", _dims.x, _dims.y);
        program.uniform1i("stencil_texture", 0);

        // Dispatch compute shader
        glDispatchCompute(_dims.x, _dims.y, 1);    
      }

      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      ASSERT_GL("Field2dCompute::compute::field()");
      TOCK_TIMER(TIMR_FIELD)
    }

    // Query field texture for positional values
    {
      TICK_TIMER(TIMR_INTERP)

      auto& program = _programs(ProgramType::eInterp);
      program.bind();
      program.uniform1ui("num_points", n);
      program.uniform2ui("texture_size", _dims.x, _dims.y);
      program.uniform1i("fields_texture", 0);

      // Bind textures and buffers
      glBindTextureUnit(0, _textures(TextureType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Dispatch compute shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute((n / 128) + 1, 1, 1);

      ASSERT_GL("Field2dCompute::compute::interp()");
      TOCK_TIMER(TIMR_INTERP)
    }
    
    // Update timers, log values on final iteration
    POLL_TIMERS()
    if (iteration == _params._iterations - 1) {
      utils::secureLog(_logger, "\nField computation");
      LOG_TIMER(_logger, TIMR_STENCIL, "  Stencil");
      LOG_TIMER(_logger, TIMR_FIELD, "  Field");
      LOG_TIMER(_logger, TIMR_INTERP, "  Interp");
    }

    // Rebuild BVH once field computation time diverges from the last one by a certain value
    if (_useBvh) {
      const uint maxIters = 6;
      const double threshold = 0.01;// 0.0075; // how the @#@$@ do I determine this well
      if (_rebuildBvhOnIter && iteration >= _params._remove_exaggeration_iter) {
        _lastRebuildTime = glTimers[TIMR_FIELD].lastMicros();
        _nRebuildIters = 0;
        _rebuildBvhOnIter = false;
      } else if (_lastRebuildTime > 0.0) {
        const double difference = std::abs(glTimers[TIMR_FIELD].lastMicros() - _lastRebuildTime) 
                                / _lastRebuildTime;
        if (difference >= threshold || ++_nRebuildIters >= maxIters) {
          _rebuildBvhOnIter = true;
        }
      }
    }
  }
}