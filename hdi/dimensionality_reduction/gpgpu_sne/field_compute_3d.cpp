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
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_shaders_3d_dual.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_3d.h"

// #define SKIP_SMALL_GRID   // Skip computation of active pixels for tiny fields, only for voxel grid computation

namespace hdi::dr {
  // Magic numbers
  constexpr float pointSize = 3.f;                          // Point size in voxel grid
  constexpr uint bvhRebuildIters = 5;                       // Nr. of iterations where the embedding hierarchy is refitted
  constexpr uint pairsInputBufferSize = 2048 * 1024 * 1024;  // Initial pair queue size in Mb. There are two such queues.
  constexpr uint pairsLeafBufferSize = 2048 * 1024 * 1024;    // Initial leaf queue size in Mb. There is one such queue.
  constexpr AlignedVec<3, unsigned> fieldBvhDims(128);      // Initial fieldBvh dims. Expansion isn't cheap.
  constexpr uint fieldHierarchyStartLvl = 2;
  constexpr uint embeddingHierarchyStartLvl = 2;

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

  // Node pairs re-used at start of dual hierarchy traversal
  constexpr auto initPairs = []() {
    const uint fBegin = (0x24924924u >> (32u - BVH_LOGK_3D * fieldHierarchyStartLvl));
    const uint fNodes = 1u << (BVH_LOGK_3D * fieldHierarchyStartLvl);
    const uint eBegin = (0x24924924u >> (32u - BVH_LOGK_3D * embeddingHierarchyStartLvl));
    const uint eNodes = 1u << (BVH_LOGK_3D * embeddingHierarchyStartLvl);
    std::array<glm::uvec2, fNodes * eNodes> a {};
    for (uint i = 0; i < fNodes; ++i) {
      for (uint j = 0; j < eNodes; ++j) {
        a[i * eNodes + j] = glm::uvec2(fBegin + i, eBegin + j);
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
    _nPixels(0),
    _useVoxelGrid(false),
    _useEmbeddingBvh(false),
    _useFieldBvh(false),
    _bvhRebuildIters(0) { }

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
                            GLuint positionBuffer,
                            GLuint boundsBuffer, 
                            unsigned n) {
    _params = params;
    _useEmbeddingBvh = _params.singleHierarchyTheta > 0.0;
    _useFieldBvh = _params.dualHierarchyTheta > 0.0;
    _useVoxelGrid = !_useEmbeddingBvh && !_useFieldBvh;

    constexpr uint kNode = BVH_KNODE_3D;
    constexpr uint logk = BVH_LOGK_3D;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }

      _programs(ProgramType::eGrid).addShader(VERTEX, grid_vert_src);
      _programs(ProgramType::eGrid).addShader(FRAGMENT, grid_fragment_src);
      _programs(ProgramType::ePixels).addShader(COMPUTE, grid_flag_src);
      _programs(ProgramType::eFlagBvh).addShader(COMPUTE, flag_bvh_src);
      _programs(ProgramType::eDispatch).addShader(COMPUTE, divide_dispatch_src);
      _programs(ProgramType::eInterp).addShader(COMPUTE, interp_src);
      _programs(ProgramType::eField).addShader(COMPUTE, field_src); 

#ifdef EMB_BVH_WIDE_TRAVERSAL_3D
      _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_wide_src); 
#else
      _programs(ProgramType::eFieldBvh).addShader(COMPUTE, field_bvh_src); 
#endif // EMB_BVH_WIDE_TRAVERSAL_3D

      _programs(ProgramType::eFieldDual).addShader(COMPUTE, field_bvh_dual_src);
      _programs(ProgramType::eFieldDualRest).addShader(COMPUTE, field_bvh_dual_rest_src);
      _programs(ProgramType::eFieldDualLeaf).addShader(COMPUTE, field_bvh_dual_leaf_src);
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
    glNamedBufferStorage(_buffers(BufferType::ePixelsHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
    glNamedBufferStorage(_buffers(BufferType::ePixelsHeadReadback), sizeof(uint), glm::value_ptr(glm::uvec4(1)), GL_CLIENT_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);

    // Specify needed objects if the voxel grid is used for pixel flagging, i.e. no hierarchies used
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
      glVertexArrayVertexBuffer(_voxelVao, 0, positionBuffer, 0, 4 * sizeof(float));
      glEnableVertexArrayAttrib(_voxelVao, 0);
      glVertexArrayAttribFormat(_voxelVao, 0, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_voxelVao, 0, 0);
    }

    // Initialize hierarchy over embedding if any hierarchy is used
    if (_useEmbeddingBvh || _useFieldBvh) {
      _embeddingBvh.setLogger(_logger);
      _embeddingBvh.init(params, EmbeddingBVH<3>::Layout(n), positionBuffer, boundsBuffer);
      _embeddingBVHRenderer.init(_embeddingBvh, boundsBuffer);
    }

    // Initialize hierarchy over field if dual hierarchy is used
    if (_useFieldBvh) {
      // Init hierarchy for a estimated larger field texture, so it doesn't resize too often.
      _fieldHier.setLogger(_logger);     
      _fieldHier.init(params, FieldHierarchy<3>::Layout(fieldBvhDims), _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead));
      _fieldHierRenderer.init(_fieldHier, boundsBuffer);

      // Glorp all the VRAM. Hey look, it's the downside of a dual hierarchy traversal
      glNamedBufferStorage(_buffers(BufferType::ePairsInput), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutput), pairsInputBufferSize, nullptr, 0);
      // glNamedBufferStorage(_buffers(BufferType::ePairsRest), pairsInputBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeaf), pairsLeafBufferSize, nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInit), initPairs.size() * sizeof(glm::uvec2), initPairs.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInputHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsOutputHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsRestHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsLeafHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePairsInitHead), sizeof(glm::uvec3), glm::value_ptr(glm::uvec3(initPairs.size(), 1, 1)), 0);

      utils::secureLogValue(_logger, "   Field3dCompute", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    _fieldRenderer.init(&_textures(TextureType::eField));

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("Field3dCompute::init()");
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
    glDeleteFramebuffers(1, &_voxelFbo);
    glDeleteVertexArrays(1, &_voxelVao);
    for (auto& program : _programs) {
      program.destroy();
    }

    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("Field3dCompute::destr()");
    _isInit = false;
  }

  /**
   * Field3dCompute::compute()
   * 
   * Computes scalar and vector fields and then samples these at positions in positionBuffer, storing
   * sampled results in interpBuffer. Depending on approximations selected, uses a number of other
   * functions defined below.
   */
  void Field3dCompute::compute(uvec dims, unsigned iteration, unsigned n,
                                GLuint positionBuffer, GLuint boundsBuffer, GLuint interpBuffer) {
    // Rescale textures if necessary
    if (_dims != dims) {
      _dims = dims;
      _bvhRebuildIters = 0;

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
      glNamedBufferStorage(_buffers(BufferType::ePixels), product(_dims) * sizeof(uvec), nullptr, 0);\

#ifdef LOG_FIELD_RESIZE
      utils::secureLogValue(_logger, "  Resized field to", dr::to_string(_dims));
#endif
    }

    // Build BVH structure over embedding
    if (_useEmbeddingBvh) {
      _embeddingBvh.compute(_bvhRebuildIters == 0, positionBuffer, boundsBuffer);
    }

    // Generate list of pixels in the field that require computation
    if (!_useEmbeddingBvh || _bvhRebuildIters == 0) {
      compactField(n, positionBuffer, boundsBuffer);
    }

    // Clear field texture
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Read nr of flagged pixels back from host-side buffer now some time has passed
    glGetNamedBufferSubData(_buffers(BufferType::ePixelsHeadReadback), 0, sizeof(uint), &_nPixels);

    // Build hierarchy over flagged pixels if certain conditions are met
    const auto fieldHierLayout = FieldHierarchy<3>::Layout(_nPixels, _dims);
    const bool fieldBvhActive = _useFieldBvh
      && iteration >= _params.removeExaggerationIter                       // After early exaggeration phase
      && static_cast<int>(_embeddingBvh.layout().nLvls) -
         static_cast<int>(fieldHierLayout.nLvls) < DUAL_BVH_LVL_DIFFERENCE; // Trees within certain depth of each other  
    if (fieldBvhActive) {
      _fieldHier.compute(_bvhRebuildIters == 0, fieldHierLayout, _buffers(BufferType::ePixels), _buffers(BufferType::ePixelsHead), boundsBuffer);
      // Re-init fieldHier renderer if it is used
      if (_fieldHier.didResize()) {
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
      // Full O(p N) field computation
      computeField(n, iteration, positionBuffer, boundsBuffer);
    }

    // Query field texture at all embedding positions
    queryField(n, positionBuffer, boundsBuffer, interpBuffer);
    
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
      logIter(iteration, fieldBvhActive);
    }
#endif
  }

  /**
   * Field3dCompute::compactField()
   * 
   * Generate a compact list of pixels in the field which require computation. This can be performed
   * either in O(p N) or O(p log N) time, depending on the availability of a BVH over the embedding.
   */
  void Field3dCompute::compactField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eFlags);
    timer.tick();
    
    // Reset queue head
    glClearNamedBufferSubData(_buffers(BufferType::ePixelsHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    // If a BVH is available, we use this to figure out what pixels to compute.
    // Otherwise, we construct a fast voxel grid and use that.
    if (_useEmbeddingBvh || _useFieldBvh) {
      // Grab BVH structure
      const auto layout = _embeddingBvh.layout();
      const auto buffers = _embeddingBvh.buffers();
      
      // Set program uniforms
      auto &program = _programs(ProgramType::eFlagBvh);
      program.bind();
      program.uniform1ui("nPos", layout.nPos);
      program.uniform1ui("nLvls", layout.nLvls);
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
      // Compute voxel grid
      // "Single-Pass GPU Solid Voxelization for Real-Time Applications",
      // Eisemann and Decoret, 2008. Src: https://dl.acm.org/doi/10.5555/1375714.1375728
      // and use that to figure it out instead
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
        
        // Specify bound framebuffer, clear it to black and depth=1
        constexpr std::array<float, 4> c = { 0.f, 0.f, 0.f, 0.f };
        constexpr float d = 1.f;
        glClearNamedFramebufferfv(_voxelFbo, GL_COLOR, 0, c.data());
        glClearNamedFramebufferfv(_voxelFbo, GL_DEPTH, 0, &d);
        glBindFramebuffer(GL_FRAMEBUFFER, _voxelFbo);

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

      // Compact the voxel grid into BufferType::ePixels
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
    
    glAssert("Fiel3dCompute::compute::flags()");
    timer.tock();
  }

  /**
   * Field3dCompute::computeField()
   * 
   * Compute full scalar and vector fields in O(p N) time.
   */
  void Field3dCompute::computeField(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eField);
    timer.tick();

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

    glAssert("Field3dCompute::compute::field()");
    timer.tock();
  }

  /**
   * Field3dCompute::computeFieldBvh()
   * 
   * Approximate full scalar and vector fields in O(p log N) time, given a BVH is available over
   * the embedding.
   */
  void Field3dCompute::computeFieldBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    auto &timer = _timers(TimerType::eField);
    timer.tick();
    
    // Get BVH structure
    const auto layout = _embeddingBvh.layout();
    const auto buffers = _embeddingBvh.buffers();

    // Divide flag queue head by workgroup size
    {
      auto &program = _programs(ProgramType::eDispatch);
      program.bind();
  #ifdef EMB_BVH_WIDE_TRAVERSAL_3D
      program.uniform1ui("div", 256 / BVH_KNODE_3D);
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
    program.uniform1f("theta2", _params.singleHierarchyTheta * _params.singleHierarchyTheta);

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
    
    glAssert("Field3dCompute::compute::field()");
    timer.tock();
  }

  /**
   * Field3dCompute::computeFieldDualBvh()
   * 
   * Approximate scalar and vector fields in O(log p log N) time, given BVHs are available over both
   * the embedding and the field's pixels. This has a rather annoying impact on memory usage, and
   * is likely inefficient for "small" fields or embeddings of tens of thousands of points.
   */
  void Field3dCompute::computeFieldDualBvh(unsigned n, unsigned iteration, GLuint positionsBuffer, GLuint boundsBuffer) {
    // Get hierarchy data
    const auto eLayout = _embeddingBvh.layout();
    const auto eBuffers = _embeddingBvh.buffers();
    const auto fLayout = _fieldHier.layout();
    const auto fBuffers = _fieldHier.buffers();

    // Reset queue heads, travesal queue
    glCopyNamedBufferSubData(_buffers(BufferType::ePairsInit), _buffers(BufferType::ePairsInput),
      0, 0, initPairs.size() * sizeof(glm::uvec2));
    glCopyNamedBufferSubData(_buffers(BufferType::ePairsInitHead), _buffers(BufferType::ePairsInputHead),
      0, 0, sizeof(uint));
    glClearNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glClearNamedBufferSubData(_buffers(BufferType::ePairsRestHead), 
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    // Compute part of field through rotating dual-hierarchy traversal
    { // Dual
      auto &timer = _timers(TimerType::eField);
      timer.tick();

      // Bind two-sided traversal program and set uniform values
      auto &dtProgram = _programs(ProgramType::eFieldDual);
      dtProgram.bind();
      dtProgram.uniform1ui("eLvls", eLayout.nLvls);
      dtProgram.uniform1ui("fLvls", fLayout.nLvls);
      dtProgram.uniform1f("theta2", _params.dualHierarchyTheta * _params.dualHierarchyTheta);

      // Bind single-sided traversal program and set uniform values
      auto &dfProgram = _programs(ProgramType::eFieldDualRest);
      dfProgram.bind();
      dfProgram.uniform1ui("eLvls", eLayout.nLvls);
      dfProgram.uniform1ui("fLvls", fLayout.nLvls);
      dfProgram.uniform1f("theta2", _params.dualHierarchyTheta * _params.dualHierarchyTheta);

      // Bind dispatch divide program and set uniform values
      auto &dsProgram = _programs(ProgramType::eDispatch);
      dsProgram.bind();
      dsProgram.uniform1ui("div", 256 / BVH_KNODE_3D); 
      
      // Bind buffers reused below
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.pos);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.node);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, fBuffers.field);
      // Buffers 5-8 are bound/rebound below ...
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::ePairsLeaf));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffers(BufferType::ePairsLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, boundsBuffer);
      // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, _buffers(BufferType::ePairsRest));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, _buffers(BufferType::ePairsRestHead));

      // Process pair queues until empty.
      const uint minLvl = std::min(fLayout.nLvls, eLayout.nLvls);
      const uint maxLvl = std::max(fLayout.nLvls, eLayout.nLvls);
      const uint progrIter = minLvl - 2;
     /*  const uint progrIter = minLvl 
        - std::abs((int) fLayout.nLvls - (int) eLayout.nLvls) 
        - (minLvl == maxLvl ? 2 : 1); */
      const uint nIters = ceilDiv(maxLvl + minLvl, 2u) + 1u;

      /* std::cout << "dual (fLvls " << fLayout.nLvls << " eLvls " << eLayout.nLvls << ")" << '\n'; */
      TraversalState state = TraversalState::eDualSubdivide;
      for (uint iter = fieldHierarchyStartLvl; iter <= nIters; ++iter) {
        // State progression
        if (state == TraversalState::eDualSubdivide && iter == progrIter) {
          state = TraversalState::eLastDualSubdivide;
        } else if (state == TraversalState::eLastDualSubdivide) {
          state = TraversalState::ePushRest;
          break;
        } else if (state == TraversalState::ePushRest) {
          state = TraversalState::eSingleSubdivide;
        }

        // Select input, output queues for current iteration
        GLuint inputHead = (state == TraversalState::ePushRest) 
                         ? _buffers(BufferType::ePairsRestHead): _buffers(BufferType::ePairsInputHead);
        GLuint inputBuffer = (state == TraversalState::ePushRest) 
                           ? _buffers(BufferType::ePairsRest) : _buffers(BufferType::ePairsInput);
        GLuint outputHead = _buffers(BufferType::ePairsOutputHead);
        GLuint outputBuffer = _buffers(BufferType::ePairsOutput);

        // Reset output queue head
        if (state != TraversalState::ePushRest) {
          glClearNamedBufferSubData(outputHead,  GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        }

        // Divide inputHead buffer by workgroupsize
        {
          dsProgram.bind();
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputHead);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
          glDispatchCompute(1, 1, 1);
          glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        }
        
        // Perform one step down dual tree
        {
          if (state == TraversalState::eDualSubdivide || state == TraversalState::eLastDualSubdivide) {
            dtProgram.bind();
            dtProgram.uniform1ui("dhLvl", iter + 1);
          } else {
            dfProgram.bind();
          }
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, inputBuffer);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, inputHead);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, outputBuffer);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, outputHead);
          glDispatchComputeIndirect(0);
          glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // Debug output
        /* {
          uint input, output, leaf, rest;
          glGetNamedBufferSubData(inputHead, 0, sizeof(uint), &input);
          glGetNamedBufferSubData(outputHead, 0, sizeof(uint), &output);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsLeafHead), 0, sizeof(uint), &leaf);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsRestHead), 0, sizeof(uint), &rest);
          std::cout << '\t' << iter << " " << state << '\t' 
                    << input << " -> " << output << "\t(l " << leaf << ")\t(r " << rest << ")\n";
        } */

        // Swap input and output queues
        if (state != TraversalState::eLastDualSubdivide) {
          std::swap(_buffers(BufferType::ePairsInput), _buffers(BufferType::ePairsOutput));
          std::swap(_buffers(BufferType::ePairsInputHead), _buffers(BufferType::ePairsOutputHead));
        }
      }

      glAssert("Field3dCompute::compute::dt()");
      timer.tock();
    } // Dual

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

      glAssert("Field3dCompute::compute::leaf()");
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

      // Bind output image
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Dispatch shader
      glDispatchCompute(ceilDiv(fLayout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

      glAssert("Field3dCompute::compute::push()");
      timer.tock();
    } // Push
  }

  /**
   * Field3dCompute::queryField()
   * 
   * Query the scalar and vector fields at all positions in positionsBuffer, and store the results 
   * in interpBuffer.
   */
  void Field3dCompute::queryField(unsigned n, GLuint positionsBuffer, GLuint boundsBuffer, GLuint interpBuffer) {
    auto &timer = _timers(TimerType::eInterp);
    timer.tick();

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

    glAssert("Field3dCompute::compute::interp()");
    timer.tock();
  }

  /**
   * Field3dCompute::logIter()
   * 
   * Log current active method and shader runtimes over last iteration.
   */
  void Field3dCompute::logIter(uint iteration, bool fieldBvhActive) const {
    // Method name
    const std::string mtd = fieldBvhActive ? "dual tree"
                          : (_useEmbeddingBvh ? "single tree" : "full");

    // Runtimes in us
    const double fieldTime = _timers(TimerType::eField)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double leafTime = _timers(TimerType::eFieldLeaf)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double pushTime = _timers(TimerType::ePush)
      .get<GLtimer::ValueType::eLast, GLtimer::TimeScale::eMicros>();
    const double time = fieldBvhActive ? (fieldTime + leafTime + pushTime) : fieldTime;

    // Log method name, iteration, nr. of pixels flagged, and total field computation time.
    utils::secureLogValue(_logger,
      std::string("  Field (") + mtd + ") ",
      std::string("iter ") + std::to_string(iteration) 
      + std::string(" - ") + std::to_string(_nPixels) + std::string(" px") 
      + std::string(" - ") + std::to_string(time)
      + std::string(" \xE6s")
    );

    // Log the more interesting dual-hierarchy shader times 
    if (fieldBvhActive) {
      utils::secureLogValue(_logger, "    Field", 
        std::to_string(fieldTime)
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
   * Field3dCompute::logTimerAverage()
   * 
   * Log average runtimes for each shader.
   */
  void Field3dCompute::logTimerAverage() const {
    if (_useEmbeddingBvh) {
      _embeddingBvh.logTimerAverage();
    }
    if (_useFieldBvh) {
      _fieldHier.logTimerAverage();
    }

    utils::secureLog(_logger, "\nField computation");
    LOG_TIMER(_logger, TimerType::eFlags, "  Flags");
    LOG_TIMER(_logger, TimerType::eField, "  Field");
    if (_useFieldBvh) {
      LOG_TIMER(_logger, TimerType::eFieldLeaf, "  Leaf");
      LOG_TIMER(_logger, TimerType::ePush, "  Push");
    }
    LOG_TIMER(_logger, TimerType::eInterp, "  Interp");
  }
}