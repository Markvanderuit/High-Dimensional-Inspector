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
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute.h"

namespace hdi::dr {
  /**
   * GpgpuKldCompute
   * 
   * Class which computes KL-divergence over a provided embedding, and representative high-dimensional
   * joint similarity distribution for the original high-dimensional input data. Rewritten based on 
   * CPU code by N. Pezotti.
   */
  template <unsigned D>
  class GpgpuKldCompute {
    using vec = dr::AlignedVec<D, float>;
    using uvec = dr::AlignedVec<D, uint>;
  
  public:
    GpgpuKldCompute();

    float compute(const typename GpgpuSneCompute<D>::Buffers embedding,
                  const GpgpuHdCompute::Buffers distribution,
                  const TsneParameters &params);
  
    void setLogger(utils::AbstractLog* logger) {
      _logger = logger;
    }

  private:
    enum class BufferType {
      eQij,
      eKlc,
      eReduceIntermediate,
      eReduceFinal,

      Length
    };

    enum class ProgramType {
      eQij,
      eReduce,
      eKl,

      Length
    };

    enum class TimerType {
      eQij,
      eReduceQij,
      eKlc,
      eReduceKlc,

      Length
    };
    
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<ProgramType, ShaderProgram> _programs;
    EnumArray<TimerType, GLtimer> _timers;
    utils::AbstractLog* _logger;
  };
}