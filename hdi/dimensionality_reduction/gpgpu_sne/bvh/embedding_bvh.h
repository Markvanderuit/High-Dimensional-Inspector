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

#pragma once

#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_sort.h"

namespace hdi::dr {
  template <unsigned D>
  class EmbeddingBVH {
  public:
    // Wrapper object to pass tree layout data to external code
    struct Layout {
      uint nPos;        // Number of embedding points stored in BVH
      uint nNodes;      // Max nr of nodes in BVH
      uint nLvls;       // Max nr of levels in BVH
      uint nodeFanout;  // Fan-out at node level: 2, 4, 8, crazy stuff, ...
      uint leafFanout;  // Fan-out at leaf level: 1...32

      Layout()
      : nPos(0), nNodes(0), nLvls(0),  nodeFanout(0), leafFanout(0) { }

      Layout(uint nPos, uint nodeFanout, uint leafFanout) 
      : nPos(nPos), 
        nodeFanout(nodeFanout), 
        leafFanout(leafFanout) 
      {
        const uint logk = static_cast<uint>(std::log2(nodeFanout));
        nLvls = 1 + static_cast<uint>(std::ceil(std::log2(nPos) / logk));
        nNodes = 0u;
        for (uint i = 0u; i < nLvls; i++) {
          nNodes |= 1u << (logk * i); 
        }
      }
    };

    // Wrapper object to pass some buffer object handles to external code
    struct Buffers {
      GLuint node0;  // Node data part 0: center of mass (xy, z) and range size (w)
      GLuint node1;  // Node data part 1: bbox diameter (xy, z) and range begin (w)
      GLuint minb;   // Bounding boxes lower bounds
      GLuint pos;    // Sorted positional data stored in tree
    };

  public:
    EmbeddingBVH();
    ~EmbeddingBVH();

    void init(const TsneParameters &params, // T-SNE params data wrapper
              const Layout &layout,         // Tree layout data wrapper
              GLuint posBuffer,             // Handle to embedding positions buffer
              GLuint boundsBuffer);         // Handle to embedding bounds buffer
    void destr();
    void compute(bool rebuild,              // Rebuild full tree on iteration?
                 unsigned iteration,        // Iteration of gradient descent
                 GLuint posBuffer,          // Handle to embedding positions buffer
                 GLuint boundsBuffer);      // Handle to embedding bounds buffer

  private:
    enum class BufferType {
      // Work queues
/*       eSubdivQueue0,
      eSubdivQueue1,
      eSubdivQueueHead0,
      eSubdivQueueHead1,
      eSubdivQueueHeadInit, */

      // Internal tree data
      eMortonUnsorted,  // Unsorted morton codes
      eMortonSorted,    // Sorted morton codes
      eIdxSorted,       // Sorted position indices
      ePosSorted,       // Sorted position vectors
      eLeafFlag,        // Ids of what leaves have to be computed
      eLeafHead,        // Atomic head for nr of leaves in eLeafFlag
      eDispatch,        // Buffer for glDispatchComputeIndirect()

      // External accessible tree data
      eNode0,           // Node center of mass and range size
      eNode1,           // Node bbox extent and range begin idx
      eMinB,            // Node bbox minimum bounds

      Length
    };

    enum class ProgramType {
      eMorton,
      ePosSorted,
      eSubdiv,
      eDivideDispatch,
      eLeaf,
      eBbox,

      Length
    };

    bool _isInit;
    Layout _layout;
    TsneParameters _params;
    BVHSorter _sorter;
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<ProgramType, ShaderProgram> _programs;
    utils::AbstractLog* _logger;

    DECL_TIMERS(
      TIMR_MORTON,
      TIMR_SORT,
      TIMR_SUBDIV,
      TIMR_LEAF,
      TIMR_BBOX
    );

  public:     
    Layout layout() const {
      return _layout;
    }

    Buffers buffers() const {
      return {
        _buffers(BufferType::eNode0),
        _buffers(BufferType::eNode1),
        _buffers(BufferType::eMinB),
        _buffers(BufferType::ePosSorted),
      };
    }

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }
  };
}