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

#include "hdi/utils/assert_by_exception.h"
#include "hdi/dimensionality_reduction/abstract_gradient_descent_tsne.h"
#include "gpgpu_sne/depth_gpgpu_sne.h"
#include "gpgpu_sne/3d_gpgpu_sne_compute.h"

namespace hdi::dr {
  //! T-SNE gradient descent in 3D using a gpu implementation
  /*!
    T-SNE gradient descent in 3D using a gpu implementation
    \author Mark van de Ruit
  */
  class GradientDescentTSNE3D : public AbstractGradientDescentTSNE {
  public:
    GradientDescentTSNE3D();
    ~GradientDescentTSNE3D();

    void initialize(const sparse_scalar_matrix_t& probabilities,
                    data::Embedding<scalar_t>* embedding, 
                    TsneParameters params = TsneParameters()) override;

    void iterate(double mult = 1.0) override;

    Bounds3D bounds() const {
      return _gpgpu_3d_sne_compute.bounds();
    }

  private:
    double computeExaggeration();

    // Underlying implementation
    // GpgpuDepthSneCompute _gpgpu_3d_sne_compute;
    Gpgpu3dSneCompute _gpgpu_3d_sne_compute;
  };
}

#endif // GRADIENT_DESCENT_TSNE_3D_H