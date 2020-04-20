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
#include <QMatrix4x4>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "gpgpu_utils.h"
#include "3d_utils.h"

// #define FIELD_IMAGE_OUTPUT // Output field images every 100 iterations
// #define FIELD_ROTATE_VIEW // Have camera randomly rotate around embedding

namespace hdi::dr {
  class DepthFieldComputation {
  public:
    DepthFieldComputation();
    ~DepthFieldComputation();

    // Initialize gpu components for computation
    void initialize(const TsneParameters& params,
                    GLuint position_buff,
                    unsigned n);

    // Remove gpu components
    void clean();

    // Compute field and depth textures
    void compute(unsigned w, unsigned h, unsigned d,
                 float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                 Bounds3D bounds);

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }

    GLuint texture() const {
      return _textures[TEXTURE_DEPTH_SINGLE];
    }

  private:
#ifdef FIELD_ROTATE_VIEW
    void updateView();
#endif // FIELD_ROTATE_VIEW

    bool _initialized;
    int _iteration;
    unsigned _w, _h, _d;

    enum TextureType {
      // Enum values matching to textures in _textures array
      TEXTURE_DEPTH_ARRAY,
      TEXTURE_DEPTH_SINGLE,
      TEXTURE_CELLMAP,
      TEXTURE_GRID,
      TEXTURE_FIELD_3D,

      // Static enum length
      TextureTypeLength 
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_DEPTH,
      PROGRAM_GRID,
      PROGRAM_FIELD_3D,
      PROGRAM_INTERP,

      // Static enum length
      ProgramTypeLength
    };

    enum FramebufferType {
      // Enum values matching to framebuffers programs in _framebuffers array
      FBO_DEPTH,
      FBO_GRID,

      // Static enum length
      FramebufferTypeLength
    };

    GLuint _point_vao;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, FramebufferTypeLength> _framebuffers;
    std::array<GLuint, TextureTypeLength> _textures;
    std::array<uint32_t, 4 * 128> _cellData;
    QMatrix4x4 _viewMatrix;
    TsneParameters _params;
    utils::AbstractLog* _logger;

    TIMERS_DECLARE(
      TIMER_DEPTH,
      TIMER_GRID,
      TIMER_FIELD_3D,
      TIMER_INTERP
    )
  };
}