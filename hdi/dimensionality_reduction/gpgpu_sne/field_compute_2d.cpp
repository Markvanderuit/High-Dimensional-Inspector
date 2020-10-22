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
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

// #define USE_WIDE_BVH // Toggle partial warp traversal of a wide BVH, only for single tree trav.

namespace hdi::dr {
  // Magic numbers
  constexpr float pointSize = 3.f;      // Point size for stencil
  constexpr uint bvhRebuildIters = 4;   // Nr. of iterations where the embedding hierarchy is refitted

  /**
   * Field2dCompute::Field2dCompute()
   * 
   * Class constructor.
   */
  Field2dCompute::Field2dCompute()
  : _isInit(false), 
    _dims(0), 
    _logger(nullptr),
    _useBvh(false),
    _bvhRebuildIters(0) { }

  /**
   * Field2dCompute::~Field2dCompute()
   * 
   * Class destructor.
   */
  Field2dCompute::~Field2dCompute() {
    if (_isInit) {
      destr();
    }
  }

  /**
   * Field2dCompute::init()
   * 
   * Initializes GPU components and subclasses.
   */
  void Field2dCompute::init(const TsneParameters& params, 
                            GLuint position_buff,
                            GLuint bounds_buff, 
                            unsigned n) {
    _params = params;
    _useBvh = _params.singleHierarchyTheta > 0.0;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }

      _programs(ProgramType::eStencil).addShader(VERTEX, stencil_vert_src);
      _programs(ProgramType::eStencil).addShader(FRAGMENT, stencil_fragment_src);
      _programs(ProgramType::ePixels).addShader(COMPUTE, pixels_bvh_src);
      _programs(ProgramType::eDispatch).addShader(COMPUTE, divide_dispatch_src);
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

    // Generate buffer, texture, framebuffer, vertex array handles
    glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
    glCreateFramebuffers(1, &_stencilFbo);
    glCreateVertexArrays(1, &_stencilVao);
    glCreateBuffers(_buffers.size(), _buffers.data());

    // Specify buffers used
    constexpr glm::uvec4 dispatch(1);
    glNamedBufferStorage(_buffers(BufferType::ePixelsHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
    glNamedBufferStorage(_buffers(BufferType::ePixelsHeadReadback), sizeof(uint), glm::value_ptr(dispatch), GL_CLIENT_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
    
    // Specify needed objects if the stencil approach is used for pixel flagging, i.e. no hierarchies used
    if (!_useBvh) {
      // Specify framebuffer
      glNamedFramebufferTexture(_stencilFbo, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
      glNamedFramebufferDrawBuffer(_stencilFbo, GL_COLOR_ATTACHMENT0);

      // Specify vertex array
      glVertexArrayVertexBuffer(_stencilVao, 0, position_buff, 0, 2 * sizeof(float));
      glEnableVertexArrayAttrib(_stencilVao, 0);
      glVertexArrayAttribFormat(_stencilVao, 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_stencilVao, 0, 0);
    }

    // Specify textures used    
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Initialize hierarchy over embedding. Different width depending on traversal type, 16
    // works well for subgroups, 4 works well for 2 dimensions.
    if (_useBvh) {
      _bvh.setLogger(_logger);
#ifdef USE_WIDE_BVH
      _bvh.init(params, EmbeddingBVH<2>::Layout(n, 16, 4), position_buff, bounds_buff);
#else
      _bvh.init(params, EmbeddingBVH<2>::Layout(n, 4, 4), position_buff, bounds_buff);
#endif // USE_WIDE_BVH
    }

    _fieldRenderer.init(&_textures(TextureType::eField));

    INIT_TIMERS()
    ASSERT_GL("Field2dCompute::init()");
    _isInit = true;
  }

  /**
   * Field2dCompute::destr()
   * 
   * Destroys GPU components and subclasses.
   */
  void Field2dCompute::destr() {
    // Destroy renderer subcomponent
    _fieldRenderer.destr();    

    // Destroy BVH subcomponents
    if (_useBvh) {
      _bvh.destr();
    }

    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_stencilFbo);
    glDeleteVertexArrays(1, &_stencilVao);
    for (auto& program : _programs) {
      program.destroy();
    }

    DSTR_TIMERS();
    ASSERT_GL("Field2dCompute::destr()");
    _isInit = false;
  }

  /**
   * Field2Compute::compute()
   * 
   * Computes scalar and vector fields and then samples these at positions in position_buff, storing
   * sampled results in interp_buff.
   */
  void Field2dCompute::compute(uvec dims, unsigned iteration, unsigned n,
                               GLuint position_buff, GLuint bounds_buff, GLuint interp_buff) {
    // Rescale textures if necessary
    if (_dims != dims) {
      _dims = dims;
      utils::secureLogValue(_logger, "  Resizing field to", dr::to_string(_dims));

      glDeleteTextures(1, &_textures(TextureType::eField));
      glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eField));
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTextureStorage2D(_textures(TextureType::eField), 1, GL_RGBA32F, _dims.x, _dims.y);

      if (!_useBvh) {
        glDeleteTextures(1, &_textures(TextureType::eStencil));
        glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eStencil));
        glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureStorage2D(_textures(TextureType::eStencil), 1, GL_R8UI, _dims.x, _dims.y);
        glNamedFramebufferTexture(_stencilFbo, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
      }

      glDeleteBuffers(1, &_buffers(BufferType::ePixels));
      glCreateBuffers(1, &_buffers(BufferType::ePixels));
      glNamedBufferStorage(_buffers(BufferType::ePixels), product(_dims) * sizeof(uvec), nullptr, 0);
    }

    // Build BVH structure over embedding
    if (_useBvh) {
      _bvh.compute(_bvhRebuildIters == 0, iteration, position_buff, bounds_buff);

      // Rebuild BVH fully only once every X iterations
      if (iteration <= _params.removeExaggerationIter || _bvhRebuildIters >= bvhRebuildIters) {
        _bvhRebuildIters = 0;
      } else {
        _bvhRebuildIters++;
      }
    }
    
    // Generate list of pixels in the field that require computation
    compactField(n, position_buff, bounds_buff);

    // Clear field texture
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Read nr of flagged pixels back from host-side buffer now some time has passed
    uint nPixels;
    glGetNamedBufferSubData(_buffers(BufferType::ePixelsHeadReadback), 0, sizeof(uint), &nPixels);

    // Perform field computation
    if (_useBvh) {
      // O(Px Log N) field approximation
      computeFieldBvh(n, iteration, position_buff, bounds_buff);
    } else {
      // O(Px N) field computation
      computeField(n, iteration, position_buff, bounds_buff);
    }

    // Query field texture at all embedding positions
    queryField(n, position_buff, bounds_buff, interp_buff);
    
    POLL_TIMERS();
    
    if (iteration == _params.iterations - 1) {
      // Output timings after final run
      utils::secureLog(_logger, "\nField computation");
      LOG_TIMER(_logger, TIMR_STENCIL, "  Stencil");
      LOG_TIMER(_logger, TIMR_FIELD, "  Field");
      LOG_TIMER(_logger, TIMR_INTERP, "  Interp");
    } else {
      // Output nice message during runtime
      utils::secureLogValue(_logger, \
        std::string("  Field"), \
        std::string("iter ") + std::to_string(iteration) 
        + std::string(" - ") + std::to_string(nPixels) + std::string(" px") 
        + std::string(" - ") + std::to_string(glTimers[TIMR_FIELD].lastMicros())
        + std::string(" \xE6s")
      );
    }
  }

  /**
   * Field3dCompute::compactField()
   * 
   * Generate a compact list of pixels in the field which require computation. This can be performed
   * either in O(p N) or O(p log N) time, depending on the availability of a BVH over the embedding.
   */
  void Field2dCompute::compactField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_STENCIL);

    if (_useBvh) {
      // Reset queue head
      glClearNamedBufferSubData(_buffers(BufferType::ePixelsHead),
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Grab BVH structure
      const auto layout = _bvh.layout();
      const auto buffers = _bvh.buffers();

      // Set program uniforms
      auto &program = _programs(ProgramType::ePixels);
      program.bind();
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("nLvls", layout.nLvls);
      program.uniform1ui("kNode", layout.nodeFanout);
      program.uniform1ui("kLeaf", layout.leafFanout);
      program.uniform2ui("textureSize", _dims.x, _dims.y);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.minb);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePixels));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixelsHead));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_dims.x, 16u), ceilDiv(_dims.y, 16u), 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    } else {
      auto& program = _programs(ProgramType::eStencil);
      program.bind();

      // Specify and clear framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, _stencilFbo);
      constexpr std::array<float, 4> clearColor = { 0.f, 0.f, 0.f, 0.f };
      constexpr float clearDepth = 1.f;
      glClearNamedFramebufferfv(_stencilFbo, GL_COLOR, 0, clearColor.data());
      glClearNamedFramebufferfv(_stencilFbo, GL_DEPTH, 0, &clearDepth);

      // Specify viewport and point size, bind bounds buffer
      glViewport(0, 0, _dims.x, _dims.y);
      glPointSize(pointSize);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boundsBuffer);
      
      // Draw a point at each embedding position
      glBindVertexArray(_stencilVao);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Copy nr of flagged pixels from device to host-side buffer for cheaper readback later
    glCopyNamedBufferSubData(_buffers(BufferType::ePixelsHead), 
                              _buffers(BufferType::ePixelsHeadReadback), 
                              0, 0, sizeof(uint));

    ASSERT_GL("Field2dCompute::compute::stencil()");
    TOCK_TIMER(TIMR_STENCIL);
  }

  /**
   * Field3dCompute::computeField()
   * 
   * Compute full scalar and vector fields in O(p N) time.
   */
  void Field2dCompute::computeField(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_FIELD);

    // Set uniforms
    auto& program = _programs(ProgramType::eField);
    program.bind();
    program.uniform2ui("textureSize", _dims.x, _dims.y);
    program.uniform1ui("nPoints", n);
    program.uniform1i("stencilTexture", 0);

    // Bind images and textures to right units
    glBindTextureUnit(0, _textures(TextureType::eStencil));
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundsBuffer);

    // Dispatch compute shader
    glDispatchCompute(_dims.x, _dims.y, 1);    
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    ASSERT_GL("Field2dCompute::compute::field()");
    TOCK_TIMER(TIMR_FIELD);
  }

  /**
   * Field3dCompute::computeFieldBvh()
   * 
   * Approximate full scalar and vector fields in O(p log N) time, given a BVH is available over
   * the embedding.
   */
  void Field2dCompute::computeFieldBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_FIELD);

    // Get BVH structure
    const auto layout = _bvh.layout();
    const auto buffers = _bvh.buffers();

    // Divide pixel buffer head by workgroup size
    {
      auto &program = _programs(ProgramType::eDispatch);
      program.bind();
  #ifdef USE_WIDE_BVH
      program.uniform1ui("div", 256 / _bvh.layout().nodeFanout);
  #else
      program.uniform1ui("div", 256);
  #endif // USE_WIDE_BVH

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelsHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

      glDispatchCompute(1, 1, 1);
    }

    // Set uniforms
    auto& program = _programs(ProgramType::eField);
    program.bind();
    program.uniform1ui("nPos", layout.nPos);
    program.uniform1ui("nLvls", layout.nLvls);
    program.uniform1ui("kNode", layout.nodeFanout);
    program.uniform1ui("kLeaf", layout.leafFanout);
    program.uniform1f("theta2", _params.singleHierarchyTheta * _params.singleHierarchyTheta); // TODO extract and make program parameter
    program.uniform2ui("textureSize", _dims.x, _dims.y);

    // Bind images and textures to right units
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.pos);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, boundsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixels));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePixelsHead));

    // Dispatch compute shader based on Buffertype::eDispatch
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    ASSERT_GL("Field2dCompute::compute::field()");
    TOCK_TIMER(TIMR_FIELD);
  }

  /**
   * Field3dCompute::queryField()
   * 
   * Query the scalar and vector fields at all positions in positionsBuffer, and store the results 
   * in interpBuffer.
   */
  void Field2dCompute::queryField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer, GLuint interpBuffer) {
    TICK_TIMER(TIMR_INTERP)

    auto& program = _programs(ProgramType::eInterp);
    program.bind();
    program.uniform1ui("num_points", n);
    program.uniform2ui("texture_size", _dims.x, _dims.y);
    program.uniform1i("fields_texture", 0);

    // Bind textures and buffers
    glBindTextureUnit(0, _textures(TextureType::eField));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interpBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boundsBuffer);

    // Dispatch compute shader
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDispatchCompute((n / 128) + 1, 1, 1);

    ASSERT_GL("Field2dCompute::compute::interp()");
    TOCK_TIMER(TIMR_INTERP)
  }
}