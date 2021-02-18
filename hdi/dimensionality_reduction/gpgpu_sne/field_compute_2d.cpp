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
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_2d_dual.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_2d.h"

namespace hdi::dr {
  // Magic numbers
  constexpr float pointSize = 3.f;      // Point size for stencil
  constexpr uint bvhRebuildIters = 5;   // Nr. of iterations where the embedding hierarchy is refitted
  constexpr uint pairsInputBufferSize = 256 * 1024 * 1024;  // Initial pair queue size in Mb. There are two such queues.
  constexpr uint pairsLeafBufferSize = 64 * 1024 * 1024;    // Initial leaf queue size in Mb. There is one such queue.
  constexpr uint pairsDfsBufferSize = 64 * 1024 * 1024;     // Initial dfs queue size in Mb. There is one such queue.
  constexpr AlignedVec<2, unsigned> fieldBvhDims(256);      // Initial fieldBvh dims. Expansion isn't cheap.
  constexpr uint fieldHierarchyStartLvl = 2;
  constexpr uint embeddingHierarchyStartLvl = 2;

  // Node pairs re-used at start of dual hierarchy traversal
  constexpr auto initPairs = []() {
    const uint fBegin = (0x2AAAAAAA >> (31u - BVH_LOGK_2D * fieldHierarchyStartLvl));
    const uint fNodes = 1u << (BVH_LOGK_2D * fieldHierarchyStartLvl);
    const uint eBegin = (0x2AAAAAAA >> (31u - BVH_LOGK_2D * embeddingHierarchyStartLvl));
    const uint eNodes = 1u << (BVH_LOGK_2D * embeddingHierarchyStartLvl);
    std::array<glm::uvec2, fNodes * eNodes> a {};
    for (uint i = 0; i < fNodes; ++i) {
      for (uint j = 0; j < eNodes; ++j) {
        a[i * eNodes + j] = glm::uvec2(fBegin + i, eBegin + j);
      }
    }
    return a;
  }();

  /**
   * Field2dCompute::Field2dCompute()
   * 
   * Class constructor.
   */
  Field2dCompute::Field2dCompute()
  : _isInit(false), 
    _logger(nullptr),
    _dims(0), 
    _useEmbeddingBvh(false),
    _useFieldBvh(false),
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
                            GLuint positionBuffer,
                            GLuint boundsBuffer, 
                            unsigned n) {
    _params = params;
    _useEmbeddingBvh = _params.singleHierarchyTheta > 0.0;
    _useFieldBvh = _params.dualHierarchyTheta > 0.0;

    constexpr uint kNode = BVH_KNODE_2D;
    constexpr uint logk = BVH_LOGK_2D;

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
      _programs(ProgramType::eField).addShader(COMPUTE, field_src); 

#ifdef EMB_BVH_WIDE_TRAVERSAL_2D
        _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_wide_src); 
#else
        _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_src); 
#endif // EMB_BVH_WIDE_TRAVERSAL_2D

      _programs(ProgramType::eFieldDual).addShader(COMPUTE, field_bvh_dual_src);
      _programs(ProgramType::eFieldDualLeaf).addShader(COMPUTE, field_bvh_dual_leaf_src);
      _programs(ProgramType::eFieldDualDfs).addShader(COMPUTE, field_bvh_dual_dfs_src);
      _programs(ProgramType::ePush).addShader(COMPUTE, push_src); 
      
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
    if (!_useEmbeddingBvh) {
      // Specify framebuffer
      glNamedFramebufferTexture(_stencilFbo, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
      glNamedFramebufferDrawBuffer(_stencilFbo, GL_COLOR_ATTACHMENT0);

      // Specify vertex array
      glVertexArrayVertexBuffer(_stencilVao, 0, positionBuffer, 0, 2 * sizeof(float));
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

    // Initialize hierarchy over embedding if any hierarchy is used
    if (_useEmbeddingBvh || _useFieldBvh) {
      _embeddingBvh.setLogger(_logger);
      _embeddingBvh.init(params, EmbeddingBVH<2>::Layout(n), positionBuffer, boundsBuffer);
      _embeddingBVHRenderer.init(_embeddingBvh, boundsBuffer);
    }
    
    // Initialize hierarchy over field if dual hierarchy is used
    if (_useFieldBvh) {
      // Init hierarchy for a estimated larger field texture, so it doesn't resize too often.
      _fieldHier.setLogger(_logger);
      _fieldHier.init(params, FieldHierarchy<2>::Layout(fieldBvhDims), _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead));
      _fieldHierRenderer.init(_fieldHier, boundsBuffer);

      // Glorp all the VRAM. Hey look, it's the downside of a dual hierarchy traversal
      glNamedBufferStorage(_buffers(BufferType::ePairsInput), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutput), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeaf), pairsLeafBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsDfs), pairsDfsBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInputHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutputHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeafHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsDfsHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInit), initPairs.size() * sizeof(glm::uvec2), initPairs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInitHead), sizeof(glm::uvec3), glm::value_ptr(glm::uvec3(initPairs.size(), 1, 1)), 0);

      utils::secureLogValue(_logger, "   Field2dCompute", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    _fieldRenderer.init(&_textures(TextureType::eField));

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("Field2dCompute::init()");
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
    if (_useEmbeddingBvh) {
      _embeddingBVHRenderer.destr();
      _embeddingBvh.destr();
    }
    if (_useFieldBvh) {
      _fieldHierRenderer.destr();
      _fieldHier.destr();
    }

    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_stencilFbo);
    glDeleteVertexArrays(1, &_stencilVao);
    for (auto& program : _programs) {
      program.destroy();
    }

    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("Field2dCompute::destr()");
    _isInit = false;
  }

  /**
   * Field2Compute::compute()
   * 
   * Computes scalar and vector fields and then samples these at positions in positionBuffer, storing
   * sampled results in interp_buff.
   */
  void Field2dCompute::compute(uvec dims, unsigned iteration, unsigned n,
                               GLuint positionBuffer, GLuint boundsBuffer, GLuint interp_buff) {
    // Rescale textures if necessary
    if (_dims != dims) {
      _dims = dims;
      _bvhRebuildIters = 0;

      glDeleteTextures(1, &_textures(TextureType::eField));
      glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eField));
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTextureStorage2D(_textures(TextureType::eField), 1, GL_RGBA32F, _dims.x, _dims.y);

      if (!_useEmbeddingBvh) {
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

#ifdef LOG_FIELD_RESIZE
      utils::secureLogValue(_logger, "  Resized field to", dr::to_string(_dims));
#endif
    }

    // Build BVH structure over embedding
    if (_useEmbeddingBvh) {
      _embeddingBvh.compute(_bvhRebuildIters == 0, positionBuffer, boundsBuffer);
    }
    
    // Generate list of pixels in the field that require computation
    if (_bvhRebuildIters == 0) {
      compactField(n, positionBuffer, boundsBuffer);
    }

    // Clear field texture
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Read nr of flagged pixels back from host-side buffer now some time has passed
    uint nPixels;
    glGetNamedBufferSubData(_buffers(BufferType::ePixelsHeadReadback), 0, sizeof(uint), &nPixels);

    // Build hierarchy over flagged pixels if certain conditions are met
    const auto fieldHierLayout = FieldHierarchy<2>::Layout(nPixels, _dims);
    const bool fieldBvhActive = _useFieldBvh
      && iteration >= _params.removeExaggerationIter                       // After early exaggeration phase
      && static_cast<int>(_embeddingBvh.layout().nLvls) -
         static_cast<int>(fieldHierLayout.nLvls) < DUAL_BVH_LVL_DIFFERENCE; // Trees within certain depth of each other  
    if (fieldBvhActive) {
      _fieldHier.compute(_bvhRebuildIters == 0, fieldHierLayout, _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead), boundsBuffer);
      // _fieldHier.compute(true, fieldHierLayout, _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead), boundsBuffer);
      // Re-init fieldHier renderer if it is used
      if (_fieldHier.didResize() && _fieldHier.isInit()) {
        _fieldHierRenderer.destr();
        _fieldHierRenderer.init(_fieldHier, boundsBuffer);
      }
    }

    // Perform field computation
    if (fieldBvhActive) {
      // Dual hierarchy, O(log p Log N) field approximation
      computeFieldDualBvh(n, iteration, positionBuffer, boundsBuffer);
    } else if (_useEmbeddingBvh) {
      // Single hierarchy, O(p Log N) field approximation
      computeFieldBvh(n, iteration, positionBuffer, boundsBuffer);
    } else {
      // O(Px N) field computation
      computeField(n, iteration, positionBuffer, boundsBuffer);
    }

    // Query field texture at all embedding positions
    queryField(n, iteration, positionBuffer, boundsBuffer, interp_buff);
    
    // Rebuild BVH fully only once every X iterations
    if (_useEmbeddingBvh && (iteration <= _params.removeExaggerationIter || _bvhRebuildIters >= bvhRebuildIters)) {
      _bvhRebuildIters = 0;
    } else {
      _bvhRebuildIters++;
    }

    glPollTimers(_timers.size(), _timers.data());
#ifdef LOG_FIELD
    if (iteration % LOG_FIELD_ITER == 0) {
      // Output nice message during runtime
      logIter(iteration, nPixels, fieldBvhActive);
    }
#endif
  }

  /**
   * Field2dCompute::compactField()
   * 
   * Generate a compact list of pixels in the field which require computation. This can be performed
   * either in O(p N) or O(p log N) time, depending on the availability of a BVH over the embedding.
   */
  void Field2dCompute::compactField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eStencil);
    timer.tick();

    if (_useEmbeddingBvh) {
      // Reset queue head
      glClearNamedBufferSubData(_buffers(BufferType::ePixelsHead),
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Grab BVH structure
      const auto layout = _embeddingBvh.layout();
      const auto buffers = _embeddingBvh.buffers();

      // Set program uniforms
      auto &program = _programs(ProgramType::ePixels);
      program.bind();
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("nLvls", layout.nLvls);
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

    glAssert("Field2dCompute::compute::stencil()");
    timer.tock();
  }

  /**
   * Field3dCompute::computeField()
   * 
   * Compute full scalar and vector fields in O(p N) time.
   */
  void Field2dCompute::computeField(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eField);
    timer.tick();

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

    glAssert("Field2dCompute::compute::field()");
    timer.tock();
  }

  /**
   * Field3dCompute::computeFieldBvh()
   * 
   * Approximate full scalar and vector fields in O(p log N) time, given a BVH is available over
   * the embedding.
   */
  void Field2dCompute::computeFieldBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eField);
    timer.tick();

    // Get BVH structure
    const auto layout = _embeddingBvh.layout();
    const auto buffers = _embeddingBvh.buffers();

    // Divide pixel buffer head by workgroup size
    {
      auto &program = _programs(ProgramType::eDispatch);
      program.bind();
  #ifdef EMB_BVH_WIDE_TRAVERSAL_2D
      program.uniform1ui("div", 256 / BVH_KNODE_2D);
  #else
      program.uniform1ui("div", 256);
  #endif // EMB_BVH_WIDE_TRAVERSAL_2D

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelsHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

      glDispatchCompute(1, 1, 1);
    }

    // Set uniforms
    auto& program = _programs(ProgramType::eFieldBvh);
    program.bind();
    program.uniform1ui("nPos", layout.nPos);
    program.uniform1ui("nLvls", layout.nLvls);
    program.uniform1f("theta2", _params.singleHierarchyTheta * _params.singleHierarchyTheta);
    program.uniform2ui("textureSize", _dims.x, _dims.y);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.pos);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, boundsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixels));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePixelsHead));

    // Bind output image
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Dispatch compute shader based on Buffertype::eDispatch
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    glAssert("Field2dCompute::compute::field()");
    timer.tock();
  }

  /**
   * Field2dCompute::computeFieldDualBvh()
   * 
   * Approximate scalar and vector fields in O(log p log N) time, given BVHs are available over both
   * the embedding and the field's pixels. This has a rather annoying impact on memory usage, and
   * is likely inefficient for "small" fields or embeddings of tens of thousands of points.
   */
  void Field2dCompute::computeFieldDualBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    // Get hierarchy data
    const auto eLayout = _embeddingBvh.layout();
    const auto eBuffers = _embeddingBvh.buffers();
    const auto fLayout = _fieldHier.layout();
    const auto fBuffers = _fieldHier.buffers();

    // Compute part of field through rotating dual-hierarchy traversal
    { // Dual
      auto &timer = _timers(TimerType::eField);
      timer.tick();
      
      // Reset queues
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsInit), _buffers(BufferType::ePairsInput),
        0, 0, initPairs.size() * sizeof(glm::uvec2));
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsInitHead), _buffers(BufferType::ePairsInputHead),
        0, 0, sizeof(uint));
      glClearNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferSubData(_buffers(BufferType::ePairsDfsHead), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Bind traversal program and set uniform values
      auto &dtProgram = _programs(ProgramType::eFieldDual);
      dtProgram.bind();
      dtProgram.uniform1ui("eLvls", eLayout.nLvls);
      dtProgram.uniform1ui("fLvls", fLayout.nLvls);
      dtProgram.uniform1f("theta2", _params.dualHierarchyTheta * _params.dualHierarchyTheta);

      // Bind dispatch divide program and set uniform values
      auto &dsProgram = _programs(ProgramType::eDispatch);
      dsProgram.bind();
      dsProgram.uniform1ui("div", 256 / BVH_KNODE_2D); 
      
      // Bind buffers reused below
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.node);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, fBuffers.field);
      // Buffers 5-8 are bound/rebound below ...
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::ePairsLeaf));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffers(BufferType::ePairsLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffers(BufferType::ePairsDfs));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, _buffers(BufferType::ePairsDfsHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, boundsBuffer);

      // Process pair queues until empty.
      for (uint i = fieldHierarchyStartLvl; i < fLayout.nLvls - 1; i++) {
        // Reset output queue head
        glClearNamedBufferSubData(_buffers(BufferType::ePairsOutputHead),
          GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        
        // Bind dispatch divide program and divide BufferType::ePairsInputHead by workgroupsize
        dsProgram.bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsInputHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        
        // Bind dual-tree traversal program and perform one step down dual tree based on Buffertype::eDispatch
        dtProgram.bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairsInput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsInputHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsOutput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::ePairsOutputHead));
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        /* { // TODO Remove
          uint inputHead, outputHead, leafHead, dfsHead;
          glGetNamedBufferSubData(_buffers(BufferType::ePairsInputHead), 0, sizeof(uint), &inputHead);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsOutputHead), 0, sizeof(uint), &outputHead);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 0, sizeof(uint), &leafHead);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsDfsHead), 0, sizeof(uint), &dfsHead);
          std::cout << "i " << i 
                    << " inpt: " << inputHead 
                    << ", oupt: " << outputHead 
                    << ", leaf: " << leafHead 
                    << ", dfs:" << dfsHead << std::endl;
        } // TODO Remove */

        // Swap input and output queue buffer handles
        std::swap(_buffers(BufferType::ePairsInput), _buffers(BufferType::ePairsOutput));
        std::swap(_buffers(BufferType::ePairsInputHead), _buffers(BufferType::ePairsOutputHead));
      }
      
      glAssert("Field2dCompute::compute::dt()");
      timer.tock();
    } // Dual

    // Use DFS to compute forces in subtrees where one leaf was reached
   /*  { // DFS
      auto &timer = _timers(TimerType::eFieldDfs);
      timer.tick();

      // Divide values in BufferType::ePairsDfsHead and store in BufferType::eDispatch
      {
        auto &_program = _programs(ProgramType::eDispatch);
        _program.bind();
        _program.uniform1ui("div", 256); 

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsDfsHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
      }

      auto &program = _programs(ProgramType::eFieldDualDfs);
      program.bind();
      program.uniform1ui("eLvls", eLayout.nLvls);
      program.uniform1ui("fLvls", fLayout.nLvls);
      program.uniform1f("theta2", _params.singleHierarchyTheta * _params.singleHierarchyTheta);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePairsDfs));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairsDfsHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, boundsBuffer);

      // Dispatch compute shader based on Buffertype::eDispatch
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("Field2dCompute::compute::dfs()");
      timer.tock();
    } // DFS */

    // Compute forces in the remaining large leaves of the dual hierarchy
    { // Leaf
      auto &timer = _timers(TimerType::eFieldLeaf);
      timer.tick();

      // Divide values in BufferType::ePairsLeafHead and store in BufferType::eDispatch
      {
        auto &_program = _programs(ProgramType::eDispatch);
        _program.bind();
        _program.uniform1ui("div", 256 / 4); 

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsLeafHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
      }

      // Bind program
      auto &program = _programs(ProgramType::eFieldDualLeaf);
      program.bind();
      program.uniform1ui("fLvls", fLayout.nLvls);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePairsLeaf));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairsLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, boundsBuffer);

      // Dispatch compute shader based on Buffertype::eDispatch
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("Field2dCompute::compute::leaf()");
      timer.tock();
    } // Leaf

    // Gather/push down/accumulate forces in field hierarchy to a flat texture
    { // Push
      auto &timer = _timers(TimerType::ePush);
      timer.tick();

      // Bind program
      auto &program = _programs(ProgramType::ePush);
      program.bind();
      program.uniform1ui("nPixels", fLayout.nPixels);
      program.uniform1ui("fLvls", fLayout.nLvls);
      program.uniform1ui("startLvl", fieldHierarchyStartLvl);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixels));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boundsBuffer);

      // Bind output image
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Dispatch shader
      glDispatchCompute(ceilDiv(fLayout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("Field2dCompute::compute::push()");
      timer.tock();
    } // Push
  }

  /**
   * Field2dCompute::queryField()
   * 
   * Query the scalar and vector fields at all positions in positionsBuffer, and store the results 
   * in interpBuffer.
   */
  void Field2dCompute::queryField(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer, GLuint interpBuffer) {
    auto &timer = _timers(TimerType::eInterp);
    timer.tick();

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
    glDispatchCompute(ceilDiv(n, 128u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glAssert("Field2dCompute::compute::interp()");
    timer.tock();
  }

  /**
   * Field2dCompute::logIter()
   * 
   * Log current active method and shader runtimes over last iteration.
   */
  void Field2dCompute::logIter(uint iteration, uint nPixels, bool fieldBvhActive) const {
    // Method name
    const std::string mtd = fieldBvhActive ? "dual tree"
                          : (_useEmbeddingBvh ? "single tree" : "full");

    // Runtimes in us
    const double fieldTime = _timers(TimerType::eField)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double dfsTime = _timers(TimerType::eFieldDfs)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double leafTime = _timers(TimerType::eFieldLeaf)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double pushTime = _timers(TimerType::ePush)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double time = fieldBvhActive ? (fieldTime + dfsTime + leafTime + pushTime) : fieldTime;

    // Log method name, iteration, nr. of pixels flagged, and total field computation time.
    utils::secureLogValue(_logger,
      std::string("  Field (") + mtd + ") ",
      std::string("iter ") + std::to_string(iteration) 
      + std::string(" - ") + std::to_string(nPixels) + std::string(" px") 
      + std::string(" - ") + std::to_string(time) + std::string(" \xE6s") // microseconds
    );

    // Log the more interesting dual-hierarchy shader times 
    if (fieldBvhActive) {
      utils::secureLogValue(_logger, "    Field", 
        std::to_string(fieldTime)
        + std::string(" \xE6s"));
      utils::secureLogValue(_logger, "    Dfs", 
        std::to_string(dfsTime)
        + std::string(" \xE6s"));
      utils::secureLogValue(_logger, "    Leaf", 
        std::to_string(leafTime)
        + std::string(" \xE6s"));
      utils::secureLogValue(_logger, "    Push", 
        std::to_string(pushTime)
        + std::string(" \xE6s"));
    }
  }

  /**
   * Field2dCompute::logTimerAverage()
   * 
   * Log average runtimes for each shader.
   */
  void Field2dCompute::logTimerAverage() const {
    if (_useEmbeddingBvh) {
      _embeddingBvh.logTimerAverage();
    }
    if (_useFieldBvh) {
      _fieldHier.logTimerAverage();
    }

    utils::secureLog(_logger, "\nField computation");
    LOG_TIMER(_logger, TimerType::eStencil, "  Stencil");
    LOG_TIMER(_logger, TimerType::eField, "  Field");
    if (_useFieldBvh) {
      LOG_TIMER(_logger, TimerType::eFieldDfs, "  Dfs");
      LOG_TIMER(_logger, TimerType::eFieldLeaf, "  Leaf");
      LOG_TIMER(_logger, TimerType::ePush, "  Push");
    }
    LOG_TIMER(_logger, TimerType::eInterp, "  Interp");
  }
}