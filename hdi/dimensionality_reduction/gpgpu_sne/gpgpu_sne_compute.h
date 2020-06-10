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
#include <cstdlib>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/2d_field_computation.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_field_computation.h"
#include "hdi/debug/renderer/embedding.hpp"

namespace hdi::dr {
  template <unsigned D>
  class GpgpuSneCompute {
    typedef glm::vec<D, float, glm::aligned_highp> vec;
    typedef glm::vec<D, uint, glm::aligned_highp> uvec;
    typedef Bounds<D> Bounds;

  public:
    GpgpuSneCompute();
    ~GpgpuSneCompute();

    void init(const Embedding *embedding,
              const TsneParameters &params,
              const SparseMatrix &P);
    void destr();
    void compute(Embedding* embeddingPtr,
                 float exaggeration,
                 unsigned iteration,
                 float mult);

    void setLogger(utils::AbstractLog *logger) {
      _logger = logger;
      _2dFieldComputation.setLogger(logger);
      _3dFieldComputation.setLogger(logger);
    }

  private:
    enum BufferType {
      // Enums matching to storage buffers in _buffers array
      BUFF_POSITION,
      BUFF_INTERP_FIELDS,
      BUFF_SUM_Q,
      BUFF_SUM_Q_REDUCE_ADD,
      BUFF_NEIGHBOUR,
      BUFF_PROBABILITIES,
      BUFF_INDEX,
      BUFF_POSITIVE_FORCES,
      BUFF_GRADIENTS,
      BUFF_PREV_GRADIENTS,
      BUFF_GAIN,
      BUFF_BOUNDS_REDUCE_ADD,
      BUFF_BOUNDS,

      // Static enum length
      BufferTypeLength
    };

    enum ProgramType {
      // Enums matching to shader programs in _programs array
      PROG_SUM_Q,
      PROG_POSITIVE_FORCES,
      PROG_GRADIENTS,
      PROG_UPDATE,
      PROG_BOUNDS,
      PROG_CENTER,

      // Static enum length
      ProgramTypeLength
    };

    DECL_TIMERS(
      TIMR_BOUNDS,
      TIMR_SUM_Q,
      TIMR_GRADIENTS,
      TIMR_UPDATE,
      TIMR_CENTER
    );

    bool _isInit;
    Bounds _bounds;
    TsneParameters _params;
    utils::AbstractLog *_logger;
    Baseline2dFieldComputation _2dFieldComputation;
    Baseline3dFieldComputation _3dFieldComputation;
    dbg::EmbeddingRenderer _embeddingRenderer;
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
  };
}