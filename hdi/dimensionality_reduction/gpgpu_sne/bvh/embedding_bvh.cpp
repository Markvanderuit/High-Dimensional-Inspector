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

#include <algorithm>
#include <bitset>
#define _USE_MATH_DEFINES
#include <cmath>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

namespace hdi::dr {
  template <unsigned D>
  EmbeddingBVH<D>::EmbeddingBVH()
  : _isInit(false), _logger(nullptr) { }

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

    // Create program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      _programs(ProgramType::eSubdiv).addShader(COMPUTE, _2d::subdiv_src);
      _programs(ProgramType::eDivideDispatch).addShader(COMPUTE, _2d::divide_dispatch_src);

      // Blargh, swap some shaders depending on embedding dimension being 2 or 3
      if constexpr (D == 2) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _2d::morton_src);
        _programs(ProgramType::ePosSorted).addShader(COMPUTE, _2d::pos_sorted_src);
        _programs(ProgramType::eLeaf).addShader(COMPUTE, _2d::leaf_src);
        _programs(ProgramType::eBbox).addShader(COMPUTE, _2d::bbox_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eMorton).addShader(COMPUTE, _3d::morton_src);
        _programs(ProgramType::ePosSorted).addShader(COMPUTE, _3d::pos_sorted_src);
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
    utils::secureLogValue(_logger, "      fanout", _layout.nodeFanout);
    utils::secureLogValue(_logger, "      nPos", _layout.nPos);
    utils::secureLogValue(_logger, "      nLvls", _layout.nLvls);
    utils::secureLogValue(_logger, "      nNodes", _layout.nNodes);

    INIT_TIMERS();
    ASSERT_GL("EmbeddingBVH::init()");
    _isInit = true;
  }

  template <unsigned D>
  void EmbeddingBVH<D>::destr() {
    _sorter.destr();
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    DSTR_TIMERS();
    ASSERT_GL("EmbeddingBVH::destr()");
    _isInit = false;
  }

  template <unsigned D>
  void EmbeddingBVH<D>::compute(bool rebuild, 
                       unsigned iteration,
                       GLuint posBuffer,
                       GLuint boundsBuffer) {
    // Generate morton codes
    if (rebuild) {
      TICK_TIMER(TIMR_MORTON);

      auto &program = _programs(ProgramType::eMorton);
      program.bind();
      program.uniform1ui("nPoints", _layout.nPos);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMortonUnsorted));

      glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      TOCK_TIMER(TIMR_MORTON);
      ASSERT_GL("EmbeddingBVH::compute::morton()");
    } // rebuild

    // Delegate sorting of position indices over morton codes to the CUDA CUB library
    // This means we pay a cost for OpenGL-CUDA interopability, unfortunately.
    // Afterwards generate sorted list of positions based on sorted indices
    if (rebuild) {
      TICK_TIMER(TIMR_SORT);
      
      _sorter.sort(_layout.nPos, _layout.nLvls * uint(std::log2(_layout.nodeFanout))); // ohboy
      
      auto &program = _programs(ProgramType::ePosSorted);
      program.bind();
      program.uniform1ui("nPoints", _layout.nPos);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eIdxSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, posBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePosSorted));

      glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      TOCK_TIMER(TIMR_SORT);
      ASSERT_GL("EmbeddingBVH::compute::posSorted()");
    } // rebuild

/*     // Perform subdivision based on generated morton codes using work queues
    if (rebuild) {
      TICK_TIMER(TIMR_SUBDIV); 

      glClearNamedBufferSubData(_buffers(BufferType::eNode0),
        GL_RGBA32F, sizeof(glm::vec4), (_layout.nNodes - 1) * sizeof(glm::vec4), GL_RGBA, GL_FLOAT, nullptr);
      glClearNamedBufferSubData(_buffers(BufferType::eNode1),
        GL_RGBA32F, sizeof(glm::vec4), (_layout.nNodes - 1) * sizeof(glm::vec4), GL_RGBA, GL_FLOAT, nullptr);

      // Reset subdiv queue head and inital value
      glCopyNamedBufferSubData(_buffers(BufferType::eSubdivQueueHeadInit), _buffers(BufferType::eSubdivQueueHead0),
        0, 0, sizeof(uint));
      glClearNamedBufferSubData(_buffers(BufferType::eSubdivQueue0), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferSubData(_buffers(BufferType::eSubdivQueue1), 
        GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Reset leaf queue head
      glClearNamedBufferData(_buffers(BufferType::eLeafHead), 
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Bind subdivision program and set uniforms
      auto &program = _programs(ProgramType::eSubdiv);
      program.bind();
      program.uniform1ui("leafFanout", _layout.leafFanout);
      program.uniform1ui("nodeFanout", _layout.nodeFanout);

      // Bind dispatch program and set uniforms
      auto &_program = _programs(ProgramType::eDivideDispatch);
      _program.bind();
      _program.uniform1ui("div", 256 / _layout.nodeFanout); 

      // Bind buffers reused below
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eMortonSorted));
      // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::eLeafFlag));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::eLeafHead));

      // Iterate through tree levels from top to bottom
      // const uint logk = std::log2(_layout.nodeFanout);
      for (uint lvl = 0; lvl < _layout.nLvls - 1; lvl++) {
        {
          // uint head = 0;
          // glGetNamedBufferSubData(_buffers(BufferType::eSubdivQueueHead0), 0, sizeof(uint), &head);
          // std::cout << lvl << '\t' << head << '\n';
          // std::vector<uint> queue(16, 0);
          // glGetNamedBufferSubData(_buffers(BufferType::eSubdivQueue0), 0, queue.size() * sizeof(uint), queue.data());
          // for (uint i = 0; i < 16; i++) {
          //   std::cout << queue[i] << ' ';
          // }
          // std::cout << '\n';
          // std::vector<glm::vec4> queue(16, glm::vec4(0));
          // glGetNamedBufferSubData(_buffers(BufferType::eNode0), 0, queue.size() * sizeof(glm::vec4), queue.data());
          // for (uint i = 0; i < 16; i++) {
          //   std::cout << queue[i].w << ' ';
          // }
        }

        // Reset output queue head
        glClearNamedBufferData(_buffers(BufferType::eSubdivQueueHead1),
          GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

        // Bind dispatch divide program and divide BufferType::ePairsInputHead by workgroupsize
        _program.bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eSubdivQueueHead0));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

        // Bind subdiv program and perform one step down tree based on BufferType::eDispatch
        program.bind();
        program.uniform1ui("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eMortonSorted));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode0));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eSubdivQueue0));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eSubdivQueueHead0));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSubdivQueue1));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _buffers(BufferType::eSubdivQueueHead1));
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Swap input and output queue buffer handles
        using std::swap;
        swap(_buffers(BufferType::eSubdivQueue0), _buffers(BufferType::eSubdivQueue1));
        swap(_buffers(BufferType::eSubdivQueueHead0), _buffers(BufferType::eSubdivQueueHead1));
      }

      TOCK_TIMER(TIMR_SUBDIV);
      ASSERT_GL("EmbeddingBVH::compute::subdiv()");
    } // subdiv */

    // Perform subdivision based on generated morton codes
    if (rebuild) {
      TICK_TIMER(TIMR_SUBDIV); 

      // Set leaf queue head to 0
      glClearNamedBufferData(_buffers(BufferType::eLeafHead), 
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::eSubdiv);
      program.bind();
      program.uniform1ui("leafFanout", _layout.leafFanout);
      program.uniform1ui("nodeFanout", _layout.nodeFanout);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eMortonSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLeafFlag));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafHead));

      // Iterate through tree levels from top to bottom
      const uint logk = std::log2(_layout.nodeFanout);
      uint begin = 0;
      for (uint lvl = 0; lvl < _layout.nLvls - 1; lvl++) {
        const uint end = begin + (1u << (logk * lvl)) - 1;
        const uint range = 1 + end - begin;

        program.uniform1ui("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        program.uniform1ui("rangeBegin", begin);
        program.uniform1ui("rangeEnd", end);

        glDispatchCompute(ceilDiv(range, 256u / _layout.nodeFanout), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        begin = end + 1;
      }

      TOCK_TIMER(TIMR_SUBDIV);
      ASSERT_GL("EmbeddingBVH::compute::subdiv()");
    } // subdiv

    // Compute leaf data
    {
      TICK_TIMER(TIMR_LEAF);

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

      TOCK_TIMER(TIMR_LEAF);
      ASSERT_GL("EmbeddingBVH::compute::leaf()");
    }

    // Compute auxiliary data, eg. bounding boxes and Barnes-Hut stuff
    {
      TICK_TIMER(TIMR_BBOX);
      
      auto &program = _programs(ProgramType::eBbox);
      program.bind();
      program.uniform1ui("nodeFanout", _layout.nodeFanout);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMinB));

      // Iterate through levels from bottom to top.
      const uint logk = std::log2(_layout.nodeFanout);
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
      
      TOCK_TIMER(TIMR_BBOX);
      ASSERT_GL("EmbeddingBVH::compute::bbox()");
    }


    // Output timer data on final iteration
    POLL_TIMERS();
    if (iteration == _params._iterations - 1) {
#ifdef GL_TIMERS_ENABLED
      utils::secureLog(_logger, "\nPointBVH building");
#endif
      LOG_TIMER(_logger, TIMR_MORTON, "  Morton");
      LOG_TIMER(_logger, TIMR_SORT, "  Sorting");
      LOG_TIMER(_logger, TIMR_SUBDIV, "  Subdiv");
      LOG_TIMER(_logger, TIMR_LEAF, "  Leaf");
      LOG_TIMER(_logger, TIMR_BBOX, "  Bbox");
    }
  }

  // Explicit template instantiations
  template class EmbeddingBVH<2>;
  template class EmbeddingBVH<3>;
}