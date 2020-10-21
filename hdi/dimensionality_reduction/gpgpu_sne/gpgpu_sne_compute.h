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

#include <cstdlib>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/field_compute_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute.h"
#include "hdi/debug/renderer/embedding.hpp"

namespace hdi::dr {
  /**
   * GpgpuSneCompute
   * 
   * Class which performs a single step of the tSNE minimization over a provided embedding fully on
   * the GPU. The template parameter indicates support for D = 2 or D = 3 dimensional embeddings. 
   * The class delegates some computation to the subclass Field2dCompute or Field3dCompute depending 
   * on D. Rewritten based on previous version by N. Pezotti.
   */
  template <unsigned D>
  class GpgpuSneCompute {
    using Bounds = AlignedBounds<D>;
    using vec = dr::AlignedVec<D, float>;
    using uvec = dr::AlignedVec<D, uint>;

  public:
    GpgpuSneCompute();
    ~GpgpuSneCompute();

    void init(const Embedding *embedding,
              const GpgpuHdCompute::Buffers distribution,
              const TsneParameters &params);
    void destr();
    void compute(Embedding* embedding,
                 float exaggeration,
                 unsigned iteration,
                 float mult);

    void setLogger(utils::AbstractLog *logger) {
      _logger = logger;
    }

  private:
    enum class BufferType {
      ePosition,
      eBounds,
      eBoundsReduceAdd,
      eSumQ,
      eSumQReduceAdd,
      eInterpFields,
      ePositiveForces,
      eGradients,
      ePrevGradients,
      eGain,

      Length
    };

    enum class ProgramType {
      eSumQ,
      ePositiveForces,
      eGradients,
      eUpdate,
      eBounds,
      eCenter,

      Length
    };

    bool _isInit;
    Bounds _bounds;
    TsneParameters _params;
    utils::AbstractLog *_logger;
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<ProgramType, ShaderProgram> _programs;
    GpgpuHdCompute::Buffers _distribution;
    Field2dCompute _field2dCompute;
    Field3dCompute _field3dCompute;
    dbg::EmbeddingRenderer<D> _embeddingRenderer;

    DECL_TIMERS(
      TIMR_BOUNDS,
      TIMR_SUM_Q,
      TIMR_F_ATTR,
      TIMR_GRADIENTS,
      TIMR_UPDATE,
      TIMR_CENTER
    );
  };
}