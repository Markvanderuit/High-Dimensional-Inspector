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

// #define USE_BVH // Use BVH for px log n field computations

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#ifdef USE_BVH
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h" 
#include "hdi/debug/renderer/bvh.hpp" 
#endif

namespace hdi::dr {
  class Baseline3dFieldComputation {
    typedef Bounds<3> Bounds;
    typedef glm::vec<3, float, glm::aligned_highp> vec;
    typedef glm::vec<3, uint, glm::aligned_highp> uvec;

  public:
    Baseline3dFieldComputation();
    ~Baseline3dFieldComputation();
    
    void init(const TsneParameters& params,
                    GLuint position_buff,
                    GLuint bounds_buff,
                    unsigned n);
    void destr();
    void compute(uvec dims, float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                 Bounds bounds);

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
#ifdef USE_BVH
      _bvh.setLogger(logger);
#endif
    }

  private:
    bool _isInit;
    int _iteration;
    uvec _dims;

    enum class TextureType {
      eCellmap,
      eGrid,
      eField,

      Length 
    };

    enum class ProgramType {
      eGrid,
      eField,
      eInterp,

      Length
    };

    EnumArray<TextureType, GLuint> _textures;
    EnumArray<ProgramType, ShaderProgram> _programs;
    std::array<uint32_t, 4 * 128> _cellData;
    GLuint _vrao_point;
    GLuint _frbo_grid;
    TsneParameters _params;
    utils::AbstractLog* _logger;
#ifdef USE_BVH
    bvh::BVH<3> _bvh;
    dbg::BvhRenderer _renderer;
    bool _rebuildBvhOnIter;
    double _lastRebuildBvhTime;
#endif
    
    // Query timers matching to each shader
    DECL_TIMERS(
      TIMR_GRID, 
      TIMR_FIELD, 
      TIMR_INTERP
    )
  };
}