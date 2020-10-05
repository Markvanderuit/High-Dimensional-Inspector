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
  class PixelBVH {
    typedef glm::vec<D, float, glm::aligned_highp> vec;
    typedef glm::vec<D, uint, glm::aligned_highp> uvec;

  public:
    // Wrapper object to pass tree layout data to external code
    struct Layout {
      uint nPixels;       // Nr of pixels flagged for computation
      uvec dims;          // Total image resolution
      uint nNodes;        // Nr of tree nodes
      uint nLvls;         // Nr of tree levels
      uint nodeFanout;    // Tree width, 8 is a good tradeoff for memory usage and BH-approximation.

      Layout()
      : nPixels(0), dims(0), nNodes(0), nLvls(0), nodeFanout(0) { }

      Layout(uvec dims, uint nodeFanout)
      : Layout(product(dims), dims, nodeFanout) { }

      Layout(uint nPixels, uvec dims, uint nodeFanout)
      : nPixels(nPixels),
        dims(dims),
        nodeFanout(nodeFanout)
      {
        const uint logk = static_cast<uint>(std::log2(nodeFanout));
        size_t nPixelsTotal = dims.x * dims.y;
        if constexpr (D == 3) {
          nPixelsTotal *= dims.z;
        }
        nLvls = 1 + static_cast<uint>(std::ceil(std::log2(nPixelsTotal) / logk));
        nNodes = 0u;
        for (uint i = 0u; i < nLvls; i++) {
          nNodes |= 1u << (logk * i); 
        }
      }
    };

    // Wrapper object to pass some buffer object handles to external code
    struct Buffers {
      GLuint node0;     // Node data: pix/voxel min (xy, z) and range mass
      GLuint node1;     // Node data: pix/voxel extent (xy, z) and range begin
      GLuint field;     // Node data: field density (x) and gradient (yz, w)
      GLuint pixel;     // Sorted flagged pixel positions stored in tree
      GLuint leafFlags; // Flagged leaf node positions in tree
      GLuint leafHead;  // Head of leafFlags buffer
    };

  public:
    PixelBVH();
    ~PixelBVH();

    void init(const TsneParameters &params, // T-SNE params data wrapper
              const Layout &layout,         // Initial tree layout data wrapper
              GLuint pixelBuffer,           // Handle to pixel flags queue buffer
              GLuint pixelHeadBuffer);      // Hanndle to pixel flags queue head
    void destr();
    void compute(unsigned iteration,        // Iteration of gradient descent
                 const Layout &layout,      // Updated layout given new set of pixels
                 GLuint pixelBuffer,        // Handle to pixel flags queue buffe
                 GLuint pixelHeadBuffer,    // Hanndle to pixel flags queue head
                 GLuint boundsBuffer);

  private:
    enum class BufferType {
      // Internal tree data
      eMortonUnsorted,  // Unsorted morton codes
      eMortonSorted,    // Sorted morton codes
      eIdxSorted,       // Sorted position indices
      ePixelSorted,     // Sorted pixel locations
      eLeafFlag,        // Ids of what leaves have to be computed
      eLeafHead,        // Atomic head for nr of leaves in eLeafFlag
      eDispatch,        // Buffer for glDispatchComputeIndirect()

      // External accessible tree data
      eNode0,            // Node data: pix/voxel min (xy, z) and range mass
      eNode1,            // Node data: pix/voxel extent (xy, z) and range begin
      eField,            // Node data: field density (x) and gradient (yz, w)
  
      Length
    };

    enum class ProgramType {
      eMorton,
      ePixelSorted,
      eSubdiv,
      eDivideDispatch,
      eLeaf,
      eBbox,

      Length
    };

    bool _isInit;
    Layout _layout;
    uint _reservedNodes;
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
        _buffers(BufferType::eField),
        _buffers(BufferType::ePixelSorted),
        _buffers(BufferType::eLeafFlag),
        _buffers(BufferType::eLeafHead)
      };
    }

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }
  };
}