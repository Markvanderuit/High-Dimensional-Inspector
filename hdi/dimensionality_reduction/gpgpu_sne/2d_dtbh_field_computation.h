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
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "2d_bh_quadtree_computation.h"
#include "gpgpu_utils.h"
#include "2d_utils.h"
#include "glsl_tester.h"

namespace hdi::dr {
  class DualTree2dFieldComputation {
  public:
    DualTree2dFieldComputation();
    ~DualTree2dFieldComputation();

    // Initialize gpu components for computation
    void initialize(const TsneParameters& params,
                    unsigned n);

    // Remove gpu components
    void clean();

    // Compute field and depth textures
    void compute(unsigned w, unsigned h,
                 float function_support, unsigned iteration, unsigned n,
                 GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                 Bounds2D bounds);

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
      _quadtreeComputation.setLogger(logger);
    }

    GLuint texture() const {
      return _textures[TEXTURE_FIELD];
    }

  private:
    bool _initialized;
    unsigned _w, _h;

    enum BufferType {
      // Enum values matching to buffers in _buffers array
      BUFFER_TREE_INDEX,

      // Static enum length
      BufferTypeLength
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_DUAL_ROOT,
      PROGRAM_DUAL_TRAVERSE,
      PROGRAM_DUAL_REDUCE,
      PROGRAM_INTERP,

      // Static enum length
      ProgramTypeLength
    };

    enum TextureType {
      // Enum values matching to textures in _textures array
      TEXTURE_FIELD_ATLAS,
      TEXTURE_FIELD,

      // Static enum length
      TextureTypeLength 
    };
    
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, TextureTypeLength> _textures;
    TsneParameters _params;
    utils::AbstractLog* _logger;
    BarnesHut2dQuadtreeComputation _quadtreeComputation;
    GlslTester _tester; // TODO Remove testerr

    // Query timers matching to each shader
    TIMERS_DECLARE(
      TIMER_DUAL_ROOT,
      TIMER_DUAL_TRAVERSE, 
      TIMER_REDUCE,
      TIMER_INTERP
    )
  };
}