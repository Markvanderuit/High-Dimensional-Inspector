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
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"

namespace hdi::dr {
  template <unsigned D>
  EmbeddingBVH<D>::EmbeddingBVH()
  : _isInit(false),
    _iteration(0),
    _logger(nullptr) { }

  template <unsigned D>
  EmbeddingBVH<D>::~EmbeddingBVH() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void EmbeddingBVH<D>::init(const TsneParameters &params,
                          const EmbeddingBVH<D>::Layout &layout,
                          GLuint posBuffer,
                          GLuint boundsBuffer) {      
    _params = params;
    _layout = layout;
    
    constexpr uint kNode = D == 2 ? BVH_KNODE_2D : BVH_KNODE_3D;

    // Create program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      _programs(ProgramType::eDivideDispatch).addShader(COMPUTE, _2d::divide_dispatch_src);

      // Blargh, swap some shaders depending on embedding dimension being 2 or 3
      if constexpr (D == 2) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _2d::morton_src);
        _programs(ProgramType::ePosSorted).addShader(COMPUTE, _2d::pos_sorted_src);
      _programs(ProgramType::eSubdiv).addShader(COMPUTE, _2d::subdiv_src);
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _2d::leaf_src);
        _programs(ProgramType::eBbox).addShader(COMPUTE, _2d::bbox_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _3d::morton_src);
        _programs(ProgramType::ePosSorted).addShader(COMPUTE, _3d::pos_sorted_src);
      _programs(ProgramType::eSubdiv).addShader(COMPUTE, _3d::subdiv_src);
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _3d::leaf_src);
        _programs(ProgramType::eBbox).addShader(COMPUTE, _3d::bbox_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
    }

    // Create buffer objects
    {
      using vecd = dr::AlignedVec<D, float>;

      // Root node data will have total mass (nPos) initialized for top node
      std::vector<glm::vec4> nodeData(_layout.nNodes, glm::vec4(0));
      nodeData[0].w = _layout.nPos;
      glm::uvec4 head(0);

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eMortonUnsorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eIdxSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePosSorted), _layout.nPos * sizeof(vecd), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafFlag), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafHead), sizeof(uint), &head[0], 0);
      glNamedBufferStorage(_buffers(BufferType::eDispatch), 3 * sizeof(uint), &head[0], 0);
      glNamedBufferStorage(_buffers(BufferType::eNode0), _layout.nNodes * sizeof(glm::vec4), nodeData.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode1), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMinB), _layout.nNodes * sizeof(vecd), nullptr, 0);
    }

    // Pass stuff to sorter
    _sorter.init(
      _buffers(BufferType::eMortonUnsorted), 
      _buffers(BufferType::eMortonSorted), 
      _buffers(BufferType::eIdxSorted), 
      _layout.nPos
    );

    // Output tree info
    utils::secureLogValue(_logger, "   EmbeddingBVH", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    utils::secureLogValue(_logger, "      fanout", kNode);
    utils::secureLogValue(_logger, "      nPos", _layout.nPos);
    utils::secureLogValue(_logger, "      nLvls", _layout.nLvls);
    utils::secureLogValue(_logger, "      nNodes", _layout.nNodes);

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("EmbeddingBVH::init()");
    _isInit = true;
  }

  template <unsigned D>
  void EmbeddingBVH<D>::destr() {
    _sorter.destr();
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("EmbeddingBVH::destr()");
    _isInit = false;
  }

  template <unsigned D>
  void EmbeddingBVH<D>::compute(bool rebuild, 
                                GLuint posBuffer,
                                GLuint boundsBuffer) {
    if (rebuild) {
      _iteration++;
    }

    constexpr uint kNode = D == 2 ? BVH_KNODE_2D : BVH_KNODE_3D;
    constexpr uint logk = D == 2 ? BVH_LOGK_2D : BVH_LOGK_3D;

    // Generate morton codes
    if (rebuild) {
      auto &timer = _timers(TimerType::eMorton);
      timer.tick();

      auto &program = _programs(ProgramType::eMorton);
      program.bind();
      program.uniform1ui("nPoints", _layout.nPos);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMortonUnsorted));

      glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("EmbeddingBVH::compute::morton()");
      timer.tock();
    } // rebuild

    // Delegate sorting of position indices over morton codes to the CUDA CUB library
    // This means we pay a cost for OpenGL-CUDA interopability, unfortunately.
    // Afterwards generate sorted list of positions based on sorted indices
    {
      auto &timer = _timers(TimerType::eSort);
      timer.tick();

      if (rebuild) {
        _sorter.sort(_layout.nPos, _layout.nLvls * logk); // ohboy
      } // rebuild

      auto &program = _programs(ProgramType::ePosSorted);
      program.bind();
      program.uniform1ui("nPoints", _layout.nPos);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eIdxSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, posBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePosSorted));

      glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("EmbeddingBVH::compute::posSorted()");
      timer.tock();
    }

    // Perform subdivision based on generated morton codes
    if (rebuild) {
      auto &timer = _timers(TimerType::eSubdiv);
      timer.tick();

      // Set leaf queue head to 0
      glClearNamedBufferData(_buffers(BufferType::eLeafHead), 
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::eSubdiv);
      program.bind();

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

        program.uniform1ui("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        program.uniform1ui("rangeBegin", begin);
        program.uniform1ui("rangeEnd", end);

        glDispatchCompute(ceilDiv(range, 256u / kNode), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        begin = end + 1;
      }

      glAssert("EmbeddingBVH::compute::subdiv()");
      timer.tock();
    } // subdiv

    // Compute leaf data
    {
      auto &timer = _timers(TimerType::eLeaf);
      timer.tick();

      // Divide contents of BufferType::eLeafHead by workgroup size
      if (rebuild) {
        auto &program = _programs(ProgramType::eDivideDispatch);
        program.bind();
        program.uniform1ui("div", 256);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLeafHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        
        glDispatchCompute(1, 1, 1);
      } // rebuild
      
      auto &program = _programs(ProgramType::eLeaf);
      program.bind();
      program.uniform1ui("nNodes", _layout.nNodes);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMinB));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePosSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafFlag));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLeafHead));

      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("EmbeddingBVH::compute::leaf()");
      timer.tock();
    }

    // Compute auxiliary data, eg. bounding boxes and Barnes-Hut stuff
    {
      auto &timer = _timers(TimerType::eBbox);
      timer.tick();
      
      auto &program = _programs(ProgramType::eBbox);
      program.bind();

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMinB));
      /* glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, boundsBuffer); */

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
  void EmbeddingBVH<D>::logTimerAverage() const {
    utils::secureLogValue(_logger, "\nPointBVH builds", _iteration);
    LOG_TIMER(_logger, TimerType::eMorton, "  Morton");
    LOG_TIMER(_logger, TimerType::eSort, "  Sorting");
    LOG_TIMER(_logger, TimerType::eSubdiv, "  Subdiv");
    LOG_TIMER(_logger, TimerType::eLeaf, "  Leaf");
    LOG_TIMER(_logger, TimerType::eBbox, "  Bbox");
  }

  // Explicit template instantiations
  template class EmbeddingBVH<2>;
  template class EmbeddingBVH<3>;
}