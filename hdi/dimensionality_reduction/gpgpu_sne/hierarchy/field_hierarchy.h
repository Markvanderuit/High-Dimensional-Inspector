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
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"

namespace hdi::dr {
  template <unsigned D>
  class FieldHierarchy {
    using vec = dr::AlignedVec<D, float>;
    using uvec = dr::AlignedVec<D, uint>;

  public:
    struct Layout {
      uint nPixels;       // Nr of pixels flagged for computation
      uvec dims;          // Dimensional field resolution
      uint nNodes;        // Max nr of nodes in hierarchy
      uint nLvls;         // Max nr of levels in hierarchy

      Layout()
      : nPixels(0), dims(0), nNodes(0), nLvls(0) { }

      Layout(uvec dims)
      : Layout(product(dims), dims) { }

      Layout(uint nPixels, uvec dims)
      : nPixels(nPixels),
        dims(dims)
      {
        constexpr uint logk = D == 2 ? BVH_LOGK_2D : BVH_LOGK_3D;
        size_t nPos = product(dims);
        nLvls = 1 + static_cast<uint>(std::ceil(std::log2(nPos) / logk));
        nNodes = 0u;
        for (uint i = 0u; i < nLvls; i++) {
          nNodes |= 1u << (logk * i); 
        }
      }
    };

    // Wrapper object to pass some buffer object handles to external code
    struct Buffers {
      GLuint node;      // Node data: node type (2 lsb), singleton ptr (30 msb)
      GLuint field;     // Node data: field density (x) and gradient (yz, w)
    };

    FieldHierarchy();
    ~FieldHierarchy();

    void init(const TsneParameters &params, // T-SNE params data wrapper
              const Layout &layout,         // Initial tree layout data wrapper
              GLuint pixelBuffer,           // Handle to pixel flags queue buffer
              GLuint pixelHeadBuffer);      // Handle to pixel flags queue head
    void destr();
    void compute(bool rebuild,              // Rebuild full tree on iteration?
                 const Layout &layout,      // Updated layout given new set of pixels
                 GLuint pixelBuffer,        // Handle to pixel flags queue buffer
                 GLuint pixelHeadBuffer,    // Handle to pixel flags queue head
                 GLuint boundsBuffer);      // Handle to bounding box buffer

  private:
    enum class BufferType {
      eNode,             // Node data: node type (2 lsb), singleton ptr (30 msb)
      eField,            // Node data: field density (x) and gradient (yz, w)
      
      Length
    };

    enum class ProgramType {
      eLeaf,
      eReduce,

      Length
    };

    enum class TimerType {
      eLeaf,
      eReduce,
      
      Length
    };

    bool _isInit;
    uint _iteration;
    bool _didResize;
    Layout _layout;
    uint _reservedNodes;
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<ProgramType, ShaderProgram> _programs;
    EnumArray<TimerType, GLtimer> _timers;
    utils::AbstractLog* _logger;
    TsneParameters _params;

  public:   
    Layout layout() const {
      return _layout;
    }

    Buffers buffers() const {
      return {
        _buffers(BufferType::eNode),
        _buffers(BufferType::eField)
      };
    }

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }
    
    void logTimerAverage() const;

    bool didResize() const {
      return _didResize;
    }
    
    bool isInit() const {
      return _isInit;
    }
  };
} // namespace hdi::dr