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
* THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI 'z'AS IS'' AND ANY EXPRESS
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
#include <sstream>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_3d_dual.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_3d.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

// #define USE_WIDE_BVH // Toggle partial warp traversal of a wide BVH, only for single tree trav
// #define SKIP_SMALL_GRID   // Skip computation of active pixels for low resolution fields
// #define DELAY_BVH_REBUILD    // Refit BVH for rebuildDelay iterations
// #define DELAY_BVH_TRAVERSAL  // Only partially redo dual hierarchy cut for rebuildDelay iterations

namespace hdi::dr {
  // Magic numbers
  constexpr float pointSize = 3.f;                      // Point size in voxel grid
  constexpr uint rebuildDelay = 0;                      // Reuse instead of rebuild the dual tree cut for X iterations
  constexpr uint cacheLevel = 5;                        // Level at which to cache the current cut
  constexpr uint pairsInputBufferSize = 256 * 1024 * 1024;   // Initial pair queue size in Mb. There are two such queues.
  constexpr uint pairsCacheBufferSize = 8 * 1024 * 1024;    // Initial cache queue size in Mb. There is one such queue.
  constexpr uint pairsApproxBufferSize = 8 * 1024 * 1024;   // Initial accept queue size in Mb. There is one such queue.
  constexpr uint pairsLeafBufferSize = 64 * 1024 * 1024;    // Initial leaf queue size in Mb. There is one such queue.
  constexpr AlignedVec<3, unsigned> pixelBvhSize(128);               // Initial pixelBVH field dims. Expansion isn't cheap.

  // Incrementing bit flags for fast voxel grid computation
  constexpr auto cellData = []() {  
    std::array<uint, 4 * 128> a {};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 32; j++) {
        a[i * 4 * 32 + i + 4 * j] = 1u << (31 - j);
      }
    }
    return a;
  }();

  /**
   * Field3dCompute::Field3dCompute()
   * 
   * Class constructor.
   */
  Field3dCompute::Field3dCompute()
  : _isInit(false), 
    _logger(nullptr),
    _dims(0), 
    _useVoxelGrid(false),
    _usePointBvh(false),
    _usePixelBvh(false),
    _rebuildBvhOnIter(true),
    _nRebuildIters(0),
    _lastRebuildTime(0.0) { }

  /**
   * Field3dCompute::~Field3dCompute()
   * 
   * Class destructor.
   */
  Field3dCompute::~Field3dCompute() {
    if (_isInit) {
      destr();
    }
  }

  /**
   * Field3dCompute::init()
   * 
   * Initializes GPU components and subclasses.
   */
  void Field3dCompute::init(const TsneParameters& params, 
                            GLuint position_buff,
                            GLuint bounds_buff, 
                            unsigned n) {
    _params = params;
    _usePointBvh = _params._theta > 0.0;
    _usePixelBvh = _params._thetaDual > 0.0;
    _useVoxelGrid = !_usePointBvh && !_usePixelBvh;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }

      _programs(ProgramType::eGrid).addShader(VERTEX, grid_vert_src);
      _programs(ProgramType::eGrid).addShader(FRAGMENT, grid_fragment_src);
      _programs(ProgramType::ePixels).addShader(COMPUTE, grid_flag_src);
      _programs(ProgramType::eFlagBvh).addShader(COMPUTE, flag_bvh_src);
      _programs(ProgramType::eDivideDispatch).addShader(COMPUTE, divide_dispatch_src);
      _programs(ProgramType::eInterp).addShader(COMPUTE, interp_src);
      _programs(ProgramType::eField).addShader(COMPUTE, field_src); 
#ifdef USE_WIDE_BVH
      _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_wide_src); 
#else
      _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_src); 
#endif // USE_WIDE_BVH
      _programs(ProgramType::eFieldDual).addShader(COMPUTE, field_bvh_dual_src); 
      _programs(ProgramType::eFieldDualDispatch).addShader(COMPUTE, bvh_dual_dispatch_src);
      _programs(ProgramType::eLeaf).addShader(COMPUTE, field_bvh_leaf_src);
      _programs(ProgramType::eIterate).addShader(COMPUTE, iterate_src); 
      _programs(ProgramType::ePush).addShader(COMPUTE, push_src); 

      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Generate buffer, texture, framebuffer, vertex array handles
    glCreateTextures(GL_TEXTURE_1D, 1, &_textures(TextureType::eCellmap));
    glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eGrid));
    glCreateTextures(GL_TEXTURE_3D, 1, &_textures(TextureType::eField));
    glCreateFramebuffers(1, &_voxelFbo);
    glCreateVertexArrays(1, &_voxelVao);
    glCreateBuffers(_buffers.size(), _buffers.data());

    // Specify buffers used in general
    constexpr glm::uvec4 dispatch(1);
    glNamedBufferStorage(_buffers(BufferType::ePixelsHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
    glNamedBufferStorage(_buffers(BufferType::ePixelsHeadReadback), sizeof(uint), glm::value_ptr(dispatch), GL_CLIENT_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);

    if (_useVoxelGrid) {
      // Specify bit flags texture
      glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eCellmap), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureStorage1D(_textures(TextureType::eCellmap), 1, GL_RGBA32UI, 128);
      glTextureSubImage1D(_textures(TextureType::eCellmap), 0, 0, 128, GL_RGBA_INTEGER, GL_UNSIGNED_INT, cellData.data());

      // Specify framebuffer
      glNamedFramebufferTexture(_voxelFbo, GL_COLOR_ATTACHMENT0, _textures(TextureType::eGrid), 0);
      glNamedFramebufferDrawBuffer(_voxelFbo, GL_COLOR_ATTACHMENT0);

      // Specify vertex array
      glVertexArrayVertexBuffer(_voxelVao, 0, position_buff, 0, 4 * sizeof(float));
      glEnableVertexArrayAttrib(_voxelVao, 0);
      glVertexArrayAttribFormat(_voxelVao, 0, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_voxelVao, 0, 0);
    }

    if (_usePointBvh || _usePixelBvh) {
      _embeddingBvh.setLogger(_logger);
      _embeddingBvh.init(params, EmbeddingBVH<3>::Layout(n, 8, 4), position_buff, bounds_buff);
      _embeddingBVHRenderer.init(_embeddingBvh, bounds_buff);
    }

    if (_usePixelBvh) {
      // Init pixel BVH for a estimated larger field texture, so it doesn't resize too often.
      _fieldBvh.setLogger(_logger);     
      _fieldBvh.init(params, FieldBVH<3>::Layout(pixelBvhSize, 8), _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead));
      _fieldBVHRenderer.init(_fieldBvh, bounds_buff);
      _rebuildDelayIters = 0;

      // Hey look it's the downside of dual hierarchy traversals. Your VRAM is glorped.
      glNamedBufferStorage(_buffers(BufferType::ePairsInput), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutput), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsCache), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeaf), pairsLeafBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsApprox), rebuildDelay > 0  ? pairsApproxBufferSize : sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInputHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutputHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsCacheHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeafHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsApproxHead), sizeof(glm::uvec4), glm::value_ptr(dispatch), 0);

      // Stored cpu-side so we can read them back quickly. Copy over when there is time left.
      glNamedBufferStorage(_buffers(BufferType::ePairsInputHeadReadback), 
        sizeof(glm::uvec4), glm::value_ptr(dispatch), GL_CLIENT_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutputHeadReadback), 
        sizeof(glm::uvec4), glm::value_ptr(dispatch), GL_CLIENT_STORAGE_BIT);

      // Init queue pair initialization buffers. We copy this over queue 0 before dual tree traversal
      {
        // Start levels for the view/point trees
        const uint vStartLvl = 1;//2;//4;
        const uint pStartLvl = 1;//2;//2;

        // Determine nr of nodes in each tree level
        const uint kNode = _fieldBvh.layout().nodeFanout;
        const uint logk = std::log2(kNode);
        uint vBegin = 0, vNodes = 1;
        for (uint i = 0; i < vStartLvl; i++) {
          vBegin += vNodes;
          vNodes *= kNode;
        }
        uint pBegin = 0, pNodes = 1;
        for (uint i = 0; i < pStartLvl; i++) {
          pBegin += pNodes;
          pNodes *= kNode;
        }

        std::vector<glm::uvec2> initPairs;
        initPairs.reserve((vNodes - vBegin) * (pNodes - pBegin));
        for (uint i = vBegin; i < vBegin + vNodes; i++) {
          for (uint j = pBegin; j < pBegin + pNodes; j++) {
            initPairs.emplace_back(i, j);
          }
        }

        _pairsInitSize = initPairs.size() * sizeof(glm::uvec2);
        _startLvl = std::min(vStartLvl, pStartLvl);

        // Fill buffers
        const glm::uvec3 initHead(initPairs.size(), 1, 1);
        glNamedBufferStorage(_buffers(BufferType::ePairsInit), _pairsInitSize, initPairs.data(), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInitHead), sizeof(glm::uvec3), glm::value_ptr(initHead), 0);
      }

      utils::secureLogValue(_logger, "   Field3dCompute", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    _fieldRenderer.init(&_textures(TextureType::eField));

    INIT_TIMERS()
    ASSERT_GL("Field3dCompute::init()");
    _isInit = true;
  }

  /**
   * Field3dCompute::destr()
   * 
   * Destroys GPU components and subclasses.
   */
  void Field3dCompute::destr() {
    // Destroy renderer subcomponent
    _fieldRenderer.destr();

    // Destroy BVH subcomponents
    if (_usePointBvh) {
      _embeddingBVHRenderer.destr();
      _embeddingBvh.destr();
    }
    if (_usePixelBvh) {
      _fieldBVHRenderer.destr();
      _fieldBvh.destr();
    }

    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_voxelFbo);
    glDeleteVertexArrays(1, &_voxelVao);
    for (auto& program : _programs) {
      program.destroy();
    }

    DSTR_TIMERS();
    ASSERT_GL("Field3dCompute::destr()");
    _isInit = false;
  }

  /**
   * Field3dCompute::compute()
   * 
   * Computes scalar and vector fields and then samples these at positions in position_buff, storing
   * sampled results in interp_buff. Depending on approximations selected, uses a number of other
   * functions defined below.
   */
  void Field3dCompute::compute(uvec dims, unsigned iteration, unsigned n,
                                GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                                Bounds bounds) {
    // Rescale field texture as dimensions change
    if (_dims != dims) {
      _dims = dims;
      utils::secureLogValue(_logger, "  Resizing field to", dr::to_string(_dims));

      glDeleteTextures(1, &_textures(TextureType::eField));
      glCreateTextures(GL_TEXTURE_3D, 1, &_textures(TextureType::eField));
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTextureStorage3D(_textures(TextureType::eField), 1, GL_RGBA32F, _dims.x, _dims.y, _dims.z);

      if (_useVoxelGrid) {
        glDeleteTextures(1, &_textures(TextureType::eGrid));
        glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eGrid));
        glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(_textures(TextureType::eGrid), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureStorage2D(_textures(TextureType::eGrid), 1, GL_RGBA32UI, _dims.x, _dims.y);
        glNamedFramebufferTexture(_voxelFbo, GL_COLOR_ATTACHMENT0, _textures(TextureType::eGrid), 0);
      }
      
      glDeleteBuffers(1, &_buffers(BufferType::ePixels));
      glCreateBuffers(1, &_buffers(BufferType::ePixels));
      glNamedBufferStorage(_buffers(BufferType::ePixels), product(_dims) * sizeof(uvec), nullptr, 0);

      _nRebuildIters = 0;
      _rebuildDelayIters = 0;
    }

    // Build BVH structure over embedding
    if (_usePointBvh) {
      _embeddingBvh.compute(_rebuildBvhOnIter, iteration, position_buff, bounds_buff);
    }
    
    // Generate list of pixels in the field that require computation
    compactField(n, position_buff, bounds_buff);

    // Clear field texture
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Read nr of flagged pixels back from host-side buffer now some time has passed
    uint nPixels;
    glGetNamedBufferSubData(_buffers(BufferType::ePixelsHeadReadback), 0, sizeof(uint), &nPixels);

    // Build BVH structure over flagged pixels if conditions are met
    const auto fieldBvhLayout = FieldBVH<3>::Layout(nPixels, _dims, 8);
    const bool fieldBvhActive 
      = _usePixelBvh                                        // We requested dual tree bvh?
      && iteration >= _params._remove_exaggeration_iter     // After early exaggeration phase?
      && static_cast<int>(_embeddingBvh.layout().nLvls) -
         static_cast<int>(fieldBvhLayout.nLvls)  < 3;   // Trees approximately equally deep?  
    // const bool fieldBvhActive = _usePixelBvh && iteration > 250 && nPixels >= 32'768;// 65'536;
    if (fieldBvhActive) {
        _fieldBvh.compute(iteration, fieldBvhLayout, 
          _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead), bounds_buff);
    }

    // Perform field computation
    if (fieldBvhActive) {
      // O(log Px Log N) field approximation
      if (_rebuildDelayIters > 0) {
        computeFieldDualBvhPartial(n, iteration, position_buff, bounds_buff);
      } else {
        computeFieldDualBvh(n, iteration, position_buff, bounds_buff);
      }
    } else if (_usePointBvh) {
      // O(Px Log N) field approximation
      computeFieldBvh(n, iteration, position_buff, bounds_buff);
    } else {
      // O(Px N) field computation
      computeField(n, iteration, position_buff, bounds_buff);
    }

    // Query field texture at all embedding positions
    queryField(n, position_buff, bounds_buff, interp_buff);

    // Update timers, log values on final iteration
    POLL_TIMERS();    

    // Output nice message during debug
    if (iteration % 1 == 0) {
      const std::string method = fieldBvhActive  ? "dual tree"
                            : _usePointBvh ? "single tree"
                                           : "full";
      utils::secureLogValue(_logger,
        std::string("  Field (") + method + ") ",
        std::string("iter ") + std::to_string(iteration) 
        + std::string(" - ") + std::to_string(nPixels) + std::string(" px") 
        + std::string(" - ") + std::to_string(glTimers[TIMR_FIELD].lastMicros())
        + std::string(" \xE6s")
      );
    }

    if (iteration >= _params._iterations - 1) {
  #ifdef GL_TIMERS_ENABLED
      utils::secureLog(_logger, "\nField computation");
  #endif
      LOG_TIMER(_logger, TIMR_FLAGS, "  Flags");
      LOG_TIMER(_logger, TIMR_FIELD, "  Field");
      if (fieldBvhActive) {
        LOG_TIMER(_logger, TIMR_PUSH, "  Push");
      }
      LOG_TIMER(_logger, TIMR_INTERP, "  Interp");
    }
    
    // Rebuild BVH once field computation time diverges from the last one by a certain degree
    // NOTE Turns out this is not all that effective in 3D, but is very effective in 2D?!
    // if (_useBvh) {
    //   const uint maxIters = 6;
    //   const double threshold = 0.005;// 0.0075; // how the @#@$@ do I determine this well
    //   if (_rebuildBvhOnIter && iteration >= _params._remove_exaggeration_iter) {
    //     _lastRebuildTime = glTimers[TIMR_FIELD].lastMicros();
    //     _nRebuildIters = 0;
    //     _rebuildBvhOnIter = false;
    //   } else if (_lastRebuildTime > 0.0) {
    //     const double difference = std::abs(glTimers[TIMR_FIELD].lastMicros() - _lastRebuildTime) 
    //                             / _lastRebuildTime;
    //     if (difference >= threshold || ++_nRebuildIters >= maxIters) {
    //       _rebuildBvhOnIter = true;
    //     }
    //   }
    // }
  }

  /**
   * Field3dCompute::compactField()
   * 
   * Generate a compact list of pixels in the field which require computation. This can be performed
   * either in O(p N) or O(p log N) time, depending on the availability of a BVH over the embedding.
   */
  void Field3dCompute::compactField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_FLAGS);
    
    // Reset queue head
    glClearNamedBufferSubData(_buffers(BufferType::ePixelsHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    // If a BVH is available, we use this to figure out what pixels to compute.
    // Otherwise, we construct a fast voxel grid based on
    // "Single-Pass GPU Solid Voxelization for Real-Time Applications",
    // Eisemann and Decoret, 2008. Src: https://dl.acm.org/doi/10.5555/1375714.1375728
    // and use that to figure it out instead.
    if (_usePointBvh || _usePixelBvh) {
      // Grab BVH structure
      const auto layout = _embeddingBvh.layout();
      const auto buffers = _embeddingBvh.buffers();

      // Set program uniforms
      auto &program = _programs(ProgramType::eFlagBvh);
      program.bind();
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("nLvls", layout.nLvls);
      program.uniform1ui("kNode", layout.nodeFanout);
      program.uniform1ui("kLeaf", layout.leafFanout);
      program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.minb);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePixels));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixelsHead));

      // Dispatch shader, runtime O(Px log N)
      glDispatchCompute(ceilDiv(_dims.x, 32u), ceilDiv(_dims.y, 8u), _dims.z);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    } else {
      // Compute voxel grid first
  #ifdef SKIP_SMALL_GRID
      bool useGrid = dims.x * dims.y * dims.z > 65'536;
      if (!useGrid) {
        // Just fill entire image with full bits
        glm::uvec4 v(~0);
        glClearTexSubImage(_textures(TextureType::eGrid), 0, 0, 0, 0,
          dims.x, dims.y, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, glm::value_ptr(v));
      } else
  #endif
      {
        // Set program uniforms
        auto& program = _programs(ProgramType::eGrid);
        program.bind();
        program.uniform1i("cellMap", 0);
        program.uniform1f("zPadding", 0.025f);
        program.uniform1ui("zDims", std::min(128u, _dims.z));

        // Specify bound textures and buffers
        glBindTextureUnit(0, _textures(TextureType::eCellmap));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boundsBuffer);
        
        // Specify bound and cleared framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, _voxelFbo);
        constexpr std::array<float, 4> clearColor = { 0.f, 0.f, 0.f, 0.f };
        constexpr float clearDepth = 1.f;
        glClearNamedFramebufferfv(_voxelFbo, GL_COLOR, 0, clearColor.data());
        glClearNamedFramebufferfv(_voxelFbo, GL_DEPTH, 0, &clearDepth);

        // Specify logic operation, viewport and point size
        glViewport(0, 0, _dims.x, _dims.y);
        glEnable(GL_COLOR_LOGIC_OP);
        glLogicOp(GL_OR);
        glPointSize(pointSize);
      
        // Specify bound vertex array and perform draw call
        glBindVertexArray(_voxelVao);
        glDrawArrays(GL_POINTS, 0, n);

        // Clean up state a bit
        glDisable(GL_COLOR_LOGIC_OP);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);      
      }

      // Compact voxel grid into pixel queue
      {
        // Set program uniforms
        auto &program = _programs(ProgramType::ePixels);
        program.bind();
        program.uniform1ui("gridDepth", std::min(128u, _dims.z));
        program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);
        program.uniform1i("gridSampler", 0);

        // Bind voxel texture to texture unit
        glBindTextureUnit(0, _textures(TextureType::eGrid));

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixels));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::ePixelsHead));

        // Dispatch compute shader
        glDispatchCompute(ceilDiv(_dims.x, 8u), ceilDiv(_dims.y, 4u),  ceilDiv(_dims.z, 4u));
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
    }

    // Copy nr of flagged pixels from device to host-side buffer for cheaper readback later
    glCopyNamedBufferSubData(_buffers(BufferType::ePixelsHead), 
                              _buffers(BufferType::ePixelsHeadReadback), 
                              0, 0, sizeof(uint));
    
    ASSERT_GL("Fiel3dCompute::compute::flags()");
    TOCK_TIMER(TIMR_FLAGS);
  }

  /**
   * Field3dCompute::computeField()
   * 
   * Compute full scalar and vector fields in O(p N) time.
   */
  void Field3dCompute::computeField(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_FIELD);

    // Set program uniforms
    auto &program = _programs(ProgramType::eField);
    program.bind();
    program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);
    program.uniform1ui("nPoints", n);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePixels));

    // Bind output image
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Dispatch compute shader based on BufferType::ePixelsHead, one workgroup per pixel
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::ePixelsHead));
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    ASSERT_GL("Field3dCompute::compute::field()");
    TOCK_TIMER(TIMR_FIELD);
  }

  /**
   * Field3dCompute::computeFieldBvh()
   * 
   * Approximate full scalar and vector fields in O(p log N) time, given a BVH is available over
   * the embedding.
   */
  void Field3dCompute::computeFieldBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    TICK_TIMER(TIMR_FIELD);
    
    // Get BVH structure
    const auto layout = _embeddingBvh.layout();
    const auto buffers = _embeddingBvh.buffers();

    // Divide flag queue head by workgroup size
    {
      auto &program = _programs(ProgramType::eDivideDispatch);
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

    // Set program uniforms
    auto &program = _programs(ProgramType::eFieldBvh);
    program.bind();
    program.uniform3ui("textureSize", _dims.x, _dims.y, _dims.z);
    program.uniform1ui("nPos", layout.nPos);
    program.uniform1ui("nLvls", layout.nLvls);
    program.uniform1ui("kNode", layout.nodeFanout);
    program.uniform1ui("kLeaf", layout.leafFanout);
    program.uniform1f("theta2", _params._theta * _params._theta);

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
    
    ASSERT_GL("Field3dCompute::compute::field()");
    TOCK_TIMER(TIMR_FIELD);
  }

  /**
   * Field3dCompute::computeFieldDualBvh()
   * 
   * Approximate scalar and vector fields in O(log p log N) time, given BVHs are available over both
   * the embedding and the field's pixels. This has a rather annoying impact on memory usage, and
   * is likely inefficient for "small" fields or embeddings of tens of thousands of points.
   */
  void Field3dCompute::computeFieldDualBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    // Compute part of field through dual-hierarchy traversal
    {
      TICK_TIMER(TIMR_FIELD);

      // Get BVH layout and buffers
      const auto pLayout = _embeddingBvh.layout();
      const auto pBuffers = _embeddingBvh.buffers();
      const auto vLayout = _fieldBvh.layout();
      const auto vBuffers = _fieldBvh.buffers();
      
      // Reset queues
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsInit), _buffers(BufferType::ePairsInput),
        0, 0, _pairsInitSize);
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsInitHead), _buffers(BufferType::ePairsInputHead),
        0, 0, sizeof(uint));
      glClearNamedBufferSubData(_buffers(BufferType::ePairsApproxHead), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      
      // Bind traversal program and set most uniform values
      auto &program = _programs(ProgramType::eFieldDual);
      program.bind();
      program.uniform1ui("nPointLvls", pLayout.nLvls);
      program.uniform1ui("nViewLvls", vLayout.nLvls);
      program.uniform1ui("kNode", pLayout.nodeFanout);
      program.uniform1ui("kLeaf", pLayout.leafFanout);
      program.uniform1f("theta2", _params._thetaDual * _params._thetaDual);

      // Bind dispatch divide program and set all uniform values
      auto &_program = _programs(ProgramType::eFieldDualDispatch);
      _program.bind();
      _program.uniform1ui("div", 256 / _embeddingBvh.layout().nodeFanout); 
      
      // Bind buffers reused below
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.field);
      // 6-9 are bound below ...
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffers(BufferType::ePairsApprox));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffers(BufferType::ePairsApproxHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, _buffers(BufferType::ePairsLeaf));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, _buffers(BufferType::ePairsLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, _buffers(BufferType::eDispatch));

      // Process pair queues until empty
      const uint nLvls = vLayout.nLvls + pLayout.nLvls - 3; // TODO maybe make this less arbitrary?
      for (uint i = _startLvl; i < nLvls; i++) {
        // Reset output queue head
        glClearNamedBufferSubData(_buffers(BufferType::ePairsOutputHead),
          GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        
        // (Re)bind swapped input and output queue buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsInput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsInputHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::ePairsOutput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::ePairsOutputHead));
        
        // Bind dispatch divide program and divide BufferType::ePairsInputHead by workgroupsize
        _program.bind();
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        
        // Bind traversal program and perform one step down tree based on Buffertype::eDispatch
        program.bind();
        program.uniform1ui("pushAcceptQueue", rebuildDelay > 0 && i < cacheLevel && _rebuildDelayIters == 0);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Cache current output queue so we don't have to re-traverse up to this point
        if (rebuildDelay > 0 && i == cacheLevel) {
          glCopyNamedBufferSubData(_buffers(BufferType::ePairsInput), _buffers(BufferType::ePairsCache),
            0, 0, pairsCacheBufferSize);
          glCopyNamedBufferSubData(_buffers(BufferType::ePairsInputHead), _buffers(BufferType::ePairsCacheHead),
            0, 0, sizeof(uint));
        }

        // {
        //   uint head = 0;
        //   glGetNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 0, sizeof(uint), &head);
        //   std::cout << head << '\n';
        // }

        // Swap input and output queue buffer handles
        using std::swap;
        swap(_buffers(BufferType::ePairsInput), _buffers(BufferType::ePairsOutput));
        swap(_buffers(BufferType::ePairsInputHead), _buffers(BufferType::ePairsOutputHead));
        swap(_buffers(BufferType::ePairsInputHeadReadback), _buffers(BufferType::ePairsOutputHeadReadback));
      }

      // When movement is slow, delay full traversal by rebuildDelay iterations
      if (iteration >= _params._remove_exaggeration_iter - 1) {
        _rebuildDelayIters = rebuildDelay;
      }

      // Compute forces in encountered large leaf pairs
      {
        // Divide values in BufferType::ePairsLeafHead and store in BufferType::eDispatch
        {
          auto &_program = _programs(ProgramType::eDivideDispatch);
          _program.bind();
          _program.uniform1ui("div", 256 / 4); 

          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsLeafHead));
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

          glDispatchCompute(1, 1, 1);
        }

        auto &program = _programs(ProgramType::eLeaf);
        program.bind();

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pBuffers.pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsLeaf));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsLeafHead));

        // Dispatch compute shader based on Buffertype::eDispatch
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      TOCK_TIMER(TIMR_FIELD);
    }

    // Push forces down through FieldBvh
    {
      TICK_TIMER(TIMR_PUSH);

      // Get BVH data and buffers
      const auto vLayout = _fieldBvh.layout();
      const auto vBuffers = _fieldBvh.buffers();

      // Divide values in vBuffers.leafHead and store in BufferType::eDispatch
      {
        auto &program = _programs(ProgramType::eDivideDispatch);
        program.bind();
        program.uniform1ui("div", 256); 

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vBuffers.leafHead);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
      }

      auto &program = _programs(ProgramType::ePush);
      program.bind();
      program.uniform1ui("kNode", vLayout.nodeFanout);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vBuffers.pixel);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.leafFlags);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.leafHead);

      // Bind output image
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

      // Dispatch compute shader based on Buffertype::eDispatch
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("Field3dCompute::compute::push()");
      TOCK_TIMER(TIMR_PUSH);
    }
  }

  /**
   * Field3dCompute::computeFieldDualBvhPartial()
   * 
   * Approximate scalar and vector fields in O(log p log N) time, given BVHs are available over both
   * the embedding and the field's pixels. This has a rather annoying impact on memory usage, and
   * is likely inefficient for "small" fields or embeddings of tens of thousands of points.
   */
  void Field3dCompute::computeFieldDualBvhPartial(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    // Compute part of field through dual-hierarchy traversal
    {
      TICK_TIMER(TIMR_FIELD);

      // Get BVH layout and buffers. The p means point tree, v means view tree (eg pixels).
      const auto pLayout = _embeddingBvh.layout();
      const auto pBuffers = _embeddingBvh.buffers();
      const auto vLayout = _fieldBvh.layout();
      const auto vBuffers = _fieldBvh.buffers();

      // Reset pair queue to previously cached pair queue
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsCache), _buffers(BufferType::ePairsInput),
        0, 0, pairsCacheBufferSize);
      glCopyNamedBufferSubData(_buffers(BufferType::ePairsCacheHead), _buffers(BufferType::ePairsInputHead),
        0, 0, sizeof(uint));
      glClearNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        
      // Revisit approximate pairs in BufferType::ePairsApprox
      {
        // Divide values in BufferType::ePairsApproxHead and store in BufferType::eDispatch
        {
          auto &_program = _programs(ProgramType::eDivideDispatch);
          _program.bind();
          _program.uniform1ui("div", 256 / 2); 

          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsApproxHead));
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

          glDispatchCompute(1, 1, 1);
        }

        auto &program = _programs(ProgramType::eIterate);
        program.bind();
        program.uniform1ui("nPointLvls", pLayout.nLvls);
        program.uniform1ui("nViewLvls", vLayout.nLvls);
        program.uniform1ui("kNode", pLayout.nodeFanout); // Assuming both are the same, are we?
        program.uniform1ui("kLeaf", pLayout.leafFanout); // Assuming both are the same, are we?
        program.uniform1f("theta2", _params._thetaDual * _params._thetaDual);

        // Bind all buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pBuffers.pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsApprox));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsApproxHead));

        // Dispatch compute shader based on Buffertype::eDispatch
        // Processes previously accepted node pairs in cache
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      // Bind traversal program and set most uniform values
      auto &program = _programs(ProgramType::eFieldDual);
      program.bind();
      program.uniform1ui("nPointLvls", pLayout.nLvls);
      program.uniform1ui("nViewLvls", vLayout.nLvls);
      program.uniform1ui("kNode", pLayout.nodeFanout); // Assuming both are the same, are we?
      program.uniform1ui("kLeaf", pLayout.leafFanout); // Assuming both are the same, are we?
      program.uniform1f("theta2", _params._thetaDual * _params._thetaDual);

      // Bind dispatch divide program and set all uniform values
      auto &_program = _programs(ProgramType::eFieldDualDispatch);
      _program.bind();
      _program.uniform1ui("div", 256 / _embeddingBvh.layout().nodeFanout); 
      
      // Bind buffers reused below
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.field);
      // 6-9 are bound below ...
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffers(BufferType::ePairsApprox));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _buffers(BufferType::ePairsApproxHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, _buffers(BufferType::ePairsLeaf));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, _buffers(BufferType::ePairsLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, _buffers(BufferType::eDispatch));


      // Process pair queues until empty
      // Start level is lower if we operate on cache of cut
      const uint startLvl = cacheLevel;
      const uint nLvls = vLayout.nLvls + pLayout.nLvls - 3; // TODO maybe make this less arbitrary?
      for (int i = startLvl; i < nLvls; i++) {
        // Reset output queue head
        glClearNamedBufferSubData(_buffers(BufferType::ePairsOutputHead),
          GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        
        // (Re)bind swapped input and output queue buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsInput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsInputHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::ePairsOutput));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::ePairsOutputHead));
        
        // Bind dispatch divide program and divide BufferType::ePairsInputHead by workgroupsize
        _program.bind();
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        
        // Bind traversal program and perform one step down tree based on Buffertype::eDispatch
        program.bind();
        program.uniform1ui("pushAcceptQueue", rebuildDelay > 0 && i < cacheLevel && _rebuildDelayIters == 0);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Swap input and output queue buffer handles
        using std::swap;
        swap(_buffers(BufferType::ePairsInput), _buffers(BufferType::ePairsOutput));
        swap(_buffers(BufferType::ePairsInputHead), _buffers(BufferType::ePairsOutputHead));
        swap(_buffers(BufferType::ePairsInputHeadReadback), _buffers(BufferType::ePairsOutputHeadReadback));
      }

      // Decrease number of iterations to delay rebuild by 1
      _rebuildDelayIters--;

      // Compute forces in encountered large leaf pairs
      {
        // Divide values in BufferType::ePairsLeafHead and store in BufferType::eDispatch
        {
          auto &_program = _programs(ProgramType::eDivideDispatch);
          _program.bind();
          _program.uniform1ui("div", 256 / 4); 

          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsLeafHead));
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

          glDispatchCompute(1, 1, 1);
        }

        auto &program = _programs(ProgramType::eLeaf);
        program.bind();

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pBuffers.pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::ePairsLeaf));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsLeafHead));

        // Dispatch compute shader based on Buffertype::eDispatch
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      TOCK_TIMER(TIMR_FIELD);
    }

    // Push forces down through FieldBvh
    {
      TICK_TIMER(TIMR_PUSH);

      // Clear field texture
      glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

      // Get BVH data and buffers
      const auto vLayout = _fieldBvh.layout();
      const auto vBuffers = _fieldBvh.buffers();

      // Divide values in vBuffers.leafHead and store in BufferType::eDispatch
      {
        auto &program = _programs(ProgramType::eDivideDispatch);
        program.bind();
        program.uniform1ui("div", 256); 

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vBuffers.leafHead);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
      }

      auto &program = _programs(ProgramType::ePush);
      program.bind();
      program.uniform1ui("kNode", vLayout.nodeFanout);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vBuffers.pixel);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, vBuffers.leafFlags);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vBuffers.leafHead);

      // Bind output image
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

      // Dispatch compute shader based on Buffertype::eDispatch
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("Field3dCompute::compute::push()");
      TOCK_TIMER(TIMR_PUSH);
    }
  }

  /**
   * Field3dCompute::queryField()
   * 
   * Query the scalar and vector fields at all positions in positionsBuffer, and store the results 
   * in interpBuffer.
   */
  void Field3dCompute::queryField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer, GLuint interpBuffer) {
    TICK_TIMER(TIMR_INTERP);

    auto &program = _programs(ProgramType::eInterp);
    program.bind();
    program.uniform1ui("nPoints", n);
    program.uniform3ui("fieldSize", _dims.x, _dims.y, _dims.z);
    program.uniform1i("fieldSampler", 0);
 
    // Bind textures and buffers for this shader
    glBindTextureUnit(0, _textures(TextureType::eField));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interpBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boundsBuffer);

    // Dispatch compute shader
    glDispatchCompute(ceilDiv(n, 128u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    ASSERT_GL("Field3dCompute::compute::interp()");
    TOCK_TIMER(TIMR_INTERP);
  }
}