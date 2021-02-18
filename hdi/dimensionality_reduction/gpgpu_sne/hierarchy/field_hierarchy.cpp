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
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy_shaders_3d.h"

namespace hdi::dr {
  template <unsigned D>
  FieldHierarchy<D>::FieldHierarchy()
  : _reservedNodes(0),
    _iteration(0),
    _isInit(false),
    _didResize(false),
    _logger(nullptr) { }

  template <unsigned D>
  FieldHierarchy<D>::~FieldHierarchy() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void FieldHierarchy<D>::init(const TsneParameters &params, 
                               const FieldHierarchy<D>::Layout &layout,       
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
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _2d::field_hierarchy_leaf_src);
        _programs(ProgramType::eReduce).addShader(COMPUTE, _2d::field_hierarchy_reduce_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _3d::field_hierarchy_leaf_src);
        _programs(ProgramType::eReduce).addShader(COMPUTE, _3d::field_hierarchy_reduce_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
    }

    // Create buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eNode), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
    }

    // Output tree info
    utils::secureLogValue(_logger, "   FieldHier", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    utils::secureLogValue(_logger, "      fanout", kNode);
    utils::secureLogValue(_logger, "      nPixels", _layout.nPixels);
    utils::secureLogValue(_logger, "      nLvls", _layout.nLvls);
    utils::secureLogValue(_logger, "      nNodes", _layout.nNodes);

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("FieldHier::init()");
    _isInit = true;
    _didResize = false;
  }

  template <unsigned D>
  void FieldHierarchy<D>::destr() {
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("FieldHier::destr()");
    _didResize = false;
    _isInit = false;
  }

  template <unsigned D>
  void FieldHierarchy<D>::compute(bool rebuild,
                                  const FieldHierarchy<D>::Layout &layout, 
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
    
    // Expand available memory for hierarchy as field grows too large
    if (rebuild && _reservedNodes < _layout.nNodes) {
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      utils::secureLog(_logger, "   Expanding FieldHier");

      // New 'reserve' layout is nearest larger power of 2, so plenty of space for now
      const uvec newDims = glm::pow(vec(2), glm::ceil(glm::log(vec(_layout.dims)) / vec(glm::log(2))));       
      FieldHierarchy<D>::Layout reserveLayout(uvec(dr::max(newDims)));
      _reservedNodes = reserveLayout.nNodes;

      // Create new buffer objects, as there's no such thing as a re-allocate, nor do the contents matter
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eNode), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);

      _didResize = true;
      utils::secureLogValue(_logger, "   Expanded FieldHie", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    // Clear hierarchy data
    if (rebuild) {
      glClearNamedBufferData(_buffers(BufferType::eNode), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    }
    glClearNamedBufferData(_buffers(BufferType::eField), GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr);

    // Generate leaf nodes
    if (rebuild) {
      auto &timer = _timers(TimerType::eLeaf);
      timer.tick();

      auto &program = _programs(ProgramType::eLeaf);
      program.bind();
      program.uniform1ui("nPixels", _layout.nPixels);
      program.uniform1ui("lvl", _layout.nLvls - 1);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pixelBuffer);

      glDispatchCompute(ceilDiv(_layout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("FieldHier::compute::leaf()");
      timer.tock();
    }

    // Reduce to form rest of nodes
    if (rebuild) {
      auto &timer = _timers(TimerType::eReduce);
      timer.tick();

      auto &program = _programs(ProgramType::eReduce);
      program.bind();

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode));
      
      for (uint lvl = _layout.nLvls - 1; lvl > 0; lvl--) {
        program.uniform1ui("lvl", lvl);
        glDispatchCompute(ceilDiv(1u << (logk * lvl), 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      glAssert("FieldHier::compute::reduce()");
      timer.tock();
    }

    glPollTimers(_timers.size(), _timers.data());
  }

  template <unsigned D>
  void FieldHierarchy<D>::logTimerAverage() const {
    utils::secureLogValue(_logger, "\nFieldHier builds", _iteration);
    LOG_TIMER(_logger, TimerType::eLeaf, "  Leaf");
    LOG_TIMER(_logger, TimerType::eReduce, "  Reduce");
  }
  
  // Explicit template instantiations
  template class FieldHierarchy<2>;
  template class FieldHierarchy<3>;
} // namespace hdi::dr