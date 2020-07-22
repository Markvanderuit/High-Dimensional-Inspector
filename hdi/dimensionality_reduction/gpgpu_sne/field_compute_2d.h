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

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"

namespace hdi::dr {
  class Field2dCompute {
    typedef Bounds<2> Bounds;
    typedef glm::vec<2, float, glm::aligned_highp> vec;
    typedef glm::vec<2, uint, glm::aligned_highp> uvec;

  public:
    Field2dCompute();
    ~Field2dCompute();

    void init(const TsneParameters& params,
                    GLuint position_buff,
                    GLuint bounds_buff,
                    unsigned n);
    void destr();
    void compute(uvec dims, 
                 float function_support, 
                 unsigned iteration, 
                 unsigned n,
                 GLuint position_buff, 
                 GLuint bounds_buff, 
                 GLuint interp_buff,
                 Bounds bounds);

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
      _bvh.setLogger(logger);
    }

  private:
    bool _isInit;
    uvec _dims;

    enum class ProgramType {
      eStencil,
      eField,
      eInterp,

      Length
    };

    enum class TextureType {
      eStencil,
      eField,

      Length
    };

    EnumArray<ProgramType, ShaderProgram> _programs;
    EnumArray<TextureType, GLuint> _textures;
    GLuint _vrao_point;
    GLuint _frbo_stencil;
    TsneParameters _params;
    utils::AbstractLog* _logger;

    // dbg::BvhRenderer _renderer;
    bool _useBvh;
    BVH<2> _bvh;
    bool _rebuildBvhOnIter;
    uint _nRebuildIters;
    double _lastRebuildTime;

    // Query timers matching to each shader
    DECL_TIMERS(
      TIMR_STENCIL, 
      TIMR_FIELD, 
      TIMR_INTERP
    );
  };
}