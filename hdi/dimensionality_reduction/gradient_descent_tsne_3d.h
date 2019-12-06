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

#ifndef GRADIENT_DESCENT_TSNE_3D_H
#define GRADIENT_DESCENT_TSNE_3D_H

#include <array>
#include <vector>
#include <cstdint>
#include "hdi/utils/assert_by_exception.h"
#include "hdi/utils/abstract_log.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "gpgpu_sne/3d_gpgpu_sne_compute.h"
#include "tsne_parameters.h"

namespace hdi::dr {
  //! T-SNE gradient descent in 3D using a gpu implementation
  /*!
    T-SNE gradient descent in 3D using a gpu implementation
    \author Mark van de Ruit
  */
  class GradientDescentTSNE3D {
  public:
    // Internal types
    typedef float scalar_t;
    typedef uint32_t data_handle_t;
    typedef data::MapMemEff<data_handle_t, scalar_t> sparse_scalar_row_t;
    typedef std::vector<sparse_scalar_row_t> sparse_scalar_matrix_t;
    typedef std::vector<scalar_t> scalar_vector_t;

    GradientDescentTSNE3D();
    ~GradientDescentTSNE3D();

    // Initialize the class with a set of distributions and parameters
    // We compute a joint probability distribution ourselves
    void initialize(const sparse_scalar_matrix_t& probabilities,
                    data::Embedding<scalar_t>* embedding, 
                    TsneParameters params = TsneParameters());

    // Perform an iteration
    void iterate(double mult = 1.0);
    
    // Compute Kullback Leibler divergence over the current generated embedding
    double computeKullbackLeiblerDivergence();

    //! Return the current logger
    utils::AbstractLog* logger() const { return _logger; }

    //! Set the current logger
    void setLogger(utils::AbstractLog* logger) { _logger = logger; }

    Gpgpu3dSneCompute::Bounds3D bounds() const {
      return _gpgpu_3d_sne_compute.bounds();
    }

    unsigned int iteration() const {
      return _iteration;
    }

  private:
    void initializeEmbedding(int seed, double mult);
    double computeExaggeration();

    bool _initialized;
    unsigned int _iteration;
    double _exaggeration_baseline;
    data::Embedding<scalar_t>* _embedding;
    sparse_scalar_matrix_t _P;
    Gpgpu3dSneCompute _gpgpu_3d_sne_compute;
    TsneParameters _params;
    utils::AbstractLog* _logger;
  };
}

#endif // GRADIENT_DESCENT_TSNE_3D_H