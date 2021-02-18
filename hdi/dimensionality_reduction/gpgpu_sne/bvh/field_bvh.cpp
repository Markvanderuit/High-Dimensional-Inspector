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

#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/field_bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/field_bvh_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/field_bvh_shaders_3d.h"

namespace hdi::dr {
  template <unsigned D>
  FieldBVH<D>::FieldBVH()
  : _isInit(false),
    _iteration(0),
    _didResize(false),
    _logger(nullptr) { }

  template <unsigned D>
  FieldBVH<D>::~FieldBVH() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void FieldBVH<D>::init(const TsneParameters &params, 
                         const FieldBVH<D>::Layout &layout,       
                         GLuint pixelBuffer,         
                         GLuint pixelHeadBuffer) {
    _params = params;
    _layout = layout;
    _reservedNodes = _layout.nNodes;

    constexpr uint kNode = D == 2 ? BVH_KNODE_2D : BVH_KNODE_3D;

    // Create program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      if constexpr (D == 2) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _2d::morton_src);
        _programs(ProgramType::ePixelSorted).addShader(COMPUTE, _2d::pixel_sorted_src);
        _programs(ProgramType::eSubdiv).addShader(COMPUTE, _2d::subdiv_src);
        _programs(ProgramType::eDivideDispatch).addShader(COMPUTE, _2d::divide_dispatch_src);
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _2d::leaf_src);
        _programs(ProgramType::eBbox).addShader(COMPUTE, _2d::bbox_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _3d::morton_src);
        _programs(ProgramType::ePixelSorted).addShader(COMPUTE, _3d::pixel_sorted_src);
        _programs(ProgramType::eSubdiv).addShader(COMPUTE, _3d::subdiv_src);
        _programs(ProgramType::eDivideDispatch).addShader(COMPUTE, _3d::divide_dispatch_src);
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _3d::leaf_src);
        _programs(ProgramType::eBbox).addShader(COMPUTE, _3d::bbox_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
    }

    // Create buffer objects
    {
      glm::uvec4 head(0);
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eMortonUnsorted), _layout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonSorted), _layout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eIdxSorted), _layout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePixelSorted), _layout.nPixels * sizeof(uvec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNode0), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNode1), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafFlag), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafHead), sizeof(uint), &head[0], 0);
      glNamedBufferStorage(_buffers(BufferType::eDispatch), 4 * sizeof(uint), &head[0], 0);
    }

    // Pass stuff to sorter
    _sorter.init(
      _buffers(BufferType::eMortonUnsorted), 
      _buffers(BufferType::eMortonSorted), 
      _buffers(BufferType::eIdxSorted), 
      _layout.nPixels
    );

    // Output tree info
    utils::secureLogValue(_logger, "   FieldBVH", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    utils::secureLogValue(_logger, "      fanout", kNode);
    utils::secureLogValue(_logger, "      nPixels", _layout.nPixels);
    utils::secureLogValue(_logger, "      nLvls", _layout.nLvls);
    utils::secureLogValue(_logger, "      nNodes", _layout.nNodes);

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("FieldBVH::init()");
    _isInit = true;
    _didResize = false;
  }

  template <unsigned D>
  void FieldBVH<D>::destr() {
    _sorter.destr();
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("FieldBVH::destr()");
    _isInit = false;
    _didResize = false;
  }

  template <unsigned D>
  void FieldBVH<D>::compute(bool rebuild,
                            const FieldBVH<D>::Layout &layout, 
                            GLuint pixelBuffer,       
                            GLuint pixelHeadBuffer,
                            GLuint boundsBuffer) {
    if (rebuild) {
      _iteration++;
    }

    _layout = layout;
    _didResize = false;

    constexpr uint kNode = D == 2 ? BVH_KNODE_2D : BVH_KNODE_3D;
    constexpr uint logk = D == 2 ? BVH_LOGK_2D : BVH_LOGK_3D;

    // Available memory for the tree becomes too small as the nr. of pixels to compute keeps
    // growing aggressively throughout the gradient descent
    if (rebuild && _reservedNodes < _layout.nNodes) {
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      utils::secureLog(_logger, "   Expanding FieldBVH");
      
      // New 'reserve' layout is nearest larger power of 2, so plenty of space for now
      const uvec newDims = glm::pow(vec(2), glm::ceil(glm::log(vec(_layout.dims)) / vec(glm::log(2))));       
      FieldBVH<D>::Layout reserveLayout(uvec(dr::max(newDims)));
      _reservedNodes = reserveLayout.nNodes;

      // Delete buffers and clear out sorder
      _sorter.destr();
      glDeleteBuffers(_buffers.size(), _buffers.data());

      // Allocate new buffers
      glm::uvec4 head(0);
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eMortonUnsorted), reserveLayout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonSorted), reserveLayout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eIdxSorted), reserveLayout.nPixels * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePixelSorted), reserveLayout.nPixels * sizeof(uvec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNode0), reserveLayout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNode1), reserveLayout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), reserveLayout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafFlag), reserveLayout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafHead), sizeof(uint), &head[0], 0);
      glNamedBufferStorage(_buffers(BufferType::eDispatch), 4 * sizeof(uint), &head[0], 0);

      // Set up new sorter using new buffers
      _sorter.init(
        _buffers(BufferType::eMortonUnsorted), 
        _buffers(BufferType::eMortonSorted), 
        _buffers(BufferType::eIdxSorted), 
        reserveLayout.nPixels
      );

      _didResize = true;
      utils::secureLogValue(_logger, "   Expanded FieldBVH", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    } // rebuild
    
    // Generate morton codes
    if (rebuild) {
      auto &timer = _timers(TimerType::eMorton);
      timer.tick();

      auto &program = _programs(ProgramType::eMorton);
      program.bind();
      program.uniform1ui("nPixels", _layout.nPixels);
      if constexpr (D == 2) {
        program.uniform2ui("textureSize", _layout.dims.x, _layout.dims.y);
      } else if constexpr (D == 3) {      
        program.uniform3ui("textureSize", _layout.dims.x, _layout.dims.y, _layout.dims.z);
      }
      
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pixelBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eMortonUnsorted));

      glDispatchCompute(ceilDiv(_layout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("FieldBVH::compute::morton()");
      timer.tock();
    } // rebuild

    // Delegate sorting of position indices over morton codes to the CUDA CUB library
    // This means we pay a cost for OpenGL-CUDA interopability, unfortunately.
    // Afterwards generate sorted list of positions based on sorted indices
    if (rebuild) {
      auto &timer = _timers(TimerType::eSort);
      timer.tick();

      _sorter.sort(_layout.nPixels, _layout.nLvls * logk);

      auto &program = _programs(ProgramType::ePixelSorted);
      program.bind();
      program.uniform1ui("nPixels", _layout.nPixels);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eIdxSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pixelBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePixelSorted));

      glDispatchCompute(ceilDiv(_layout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("FieldBVH::compute::sort()");
      timer.tock();
    } // rebuild

    // Clear out old tree nodes
    if (rebuild) {
      glClearNamedBufferData(_buffers(BufferType::eNode0),
        GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eNode1),
        GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Perform subdivision
    if (rebuild) {
      auto &timer = _timers(TimerType::eSubdiv);
      timer.tick();

      // Set leaf queue head to 0
      glClearNamedBufferData(_buffers(BufferType::eLeafHead), 
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::eSubdiv);
      program.bind();
      program.uniform1ui("nPixels", _layout.nPixels);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eMortonSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLeafFlag));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafHead));

      // Iterate through tree levels from top to bottom
      uint begin = 0;
      for (uint lvl = 0; lvl < _layout.nLvls - 1; lvl++) {
        const uint end = begin + (1u << (logk * lvl)) - 1;
        const uint range = 1 + end - begin;

        program.uniform1ui("isTop", lvl == 0);
        program.uniform1ui("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        program.uniform1ui("rangeBegin", begin);
        program.uniform1ui("rangeEnd", end);

        glDispatchCompute(ceilDiv(range, 256u / kNode), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        begin = end + 1;
      }

      glAssert("FieldBVH::compute::subdiv()");
      timer.tock();
    } // rebuild

    // Compute leaf data
    {
      auto &timer = _timers(TimerType::eLeaf);
      timer.tick();
      
      // Divide contents of BufferType::eLeafHead by workgroup size
      {
        auto &program = _programs(ProgramType::eDivideDispatch);
        program.bind();
        program.uniform1ui("div", 256);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLeafHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        
        glDispatchCompute(1, 1, 1);
      }
      
      auto &program = _programs(ProgramType::eLeaf);
      program.bind();
      program.uniform1ui("nNodes", _layout.nNodes);
      if constexpr (D == 2) {
        program.uniform2ui("textureSize", _layout.dims.x, _layout.dims.y);
      } else if constexpr (D == 3) {      
        program.uniform3ui("textureSize", _layout.dims.x, _layout.dims.y, _layout.dims.z);
      }

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePixelSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafFlag));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLeafHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, boundsBuffer);
  
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("FieldBVH::compute::leaf()");
      timer.tock();
    }

    // Compute auxiliary data, eg. bounding boxes
    {
      auto &timer = _timers(TimerType::eBbox);
      timer.tick();
      
      auto &program = _programs(ProgramType::eBbox);
      program.bind();

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eField));

      // Iterate through levels from bottom to top.
      uint end = _layout.nNodes - 1;
      for (int lvl = _layout.nLvls - 1; lvl > 0; lvl--) {
        const uint begin = 1 + end - (1u << (logk * lvl));
        const uint range = end - begin;
                  
        program.uniform1ui("rangeBegin", begin);
        program.uniform1ui("rangeEnd", end);

        glDispatchCompute(ceilDiv(range, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        end = begin - 1;
      }

      glAssert("EmbeddingBVH::compute::bbox()");
      timer.tock();
    }

    glPollTimers(_timers.size(), _timers.data());
  }

  template <unsigned D>
  void FieldBVH<D>::logTimerAverage() const {
    utils::secureLogValue(_logger, "\nFieldBVH builds", _iteration);
    LOG_TIMER(_logger, TimerType::eMorton, "  Morton");
    LOG_TIMER(_logger, TimerType::eSort, "  Sorting");
    LOG_TIMER(_logger, TimerType::eSubdiv, "  Subdiv");
    LOG_TIMER(_logger, TimerType::eLeaf, "  Leaf");
    LOG_TIMER(_logger, TimerType::eBbox, "  Bbox");
  }
  
  // Explicit template instantiations
  template class FieldBVH<2>;
  template class FieldBVH<3>;
}