/*
*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
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
*
*/

#pragma once

#include <cstdint>
#include <vector>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute.h"

namespace hdi::dr {
  class GradientDescentTSNE {
    typedef std::vector<data::MapMemEff<uint32_t, float>> SparseMatrix;
    typedef data::Embedding<float> Embedding;

  public:
    GradientDescentTSNE();
    ~GradientDescentTSNE();

    // Lifetime management
    void init(const SparseMatrix& P,
              Embedding* embeddingPtr,
              TsneParameters params);
    void destr();

    // Iterator management
    void iterate(double mult = 1.0);

    // Misc. computation
    double computeKullbackLeiblerDivergence() const;
    double computeExaggeration() const;

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 

      _gpgpu_2d_sne_compute.setLogger(logger);
      _gpgpu_3d_sne_compute.setLogger(logger);
    }

    unsigned iteration() const
    {
      return _iteration;
    }

  private:
    bool _isInit;
    unsigned _iteration;
    double _exaggeration_baseline;
    Embedding* _embeddingPtr;
    TsneParameters _params;
    SparseMatrix _P; 
    utils::AbstractLog* _logger;

    // Underlying compute classes for 2 and 3 dimensions
    GpgpuSneCompute<2> _gpgpu_2d_sne_compute;
    GpgpuSneCompute<3> _gpgpu_3d_sne_compute;
  };
}