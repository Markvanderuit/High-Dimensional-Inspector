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

#include <vector>
#include "hdi/utils/abstract_log.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute.h"

namespace hdi::dr {
  class GpgpuTSNE {
  public:
    GpgpuTSNE();
    ~GpgpuTSNE();

    void init(const std::vector<float>& data, TsneParameters params);
    void destr();
    void iterate();

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
      _gpgpuHdCompute.setLogger(_logger);
      _gpgpu2dSneCompute.setLogger(_logger);
      _gpgpu3dSneCompute.setLogger(_logger);
    }

    std::vector<float> getRawEmbedding() const;
    float getKLDivergence();

  private:
    bool _isInit;
    unsigned _iteration;
    TsneParameters _params;
    utils::AbstractLog* _logger;

    // Underlying compute classes
    GpgpuHdCompute _gpgpuHdCompute;
    GpgpuSneCompute<2> _gpgpu2dSneCompute;
    GpgpuSneCompute<3> _gpgpu3dSneCompute;
  };
}