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

#include <cstring>
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_tsne.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_kld_compute.h"

namespace hdi::dr {
  GpgpuTSNE::GpgpuTSNE()
  : _isInit(false),
    _logger(nullptr),
    _iteration(0)
  { }
  
  GpgpuTSNE::~GpgpuTSNE()
  {
    if (_isInit) {
      destr();
    }
  }

  void GpgpuTSNE::init(const std::vector<float>& data, TsneParameters params)
  {
    _params = params;

    // Output params
    utils::secureLog(_logger, "Initializing tSNE...");
    utils::secureLogValue(_logger, "  Points", _params.n);
    utils::secureLogValue(_logger, "  High dimensions", _params.nHighDimensions);
    utils::secureLogValue(_logger, "  Low dimensions", _params.nLowDimensions);
    utils::secureLogValue(_logger, "  Perplexity", _params.perplexity);
    utils::secureLogValue(_logger, "  Iterations", _params.iterations);
    utils::secureLogValue(_logger, "  Theta (single hierarchy)", _params.singleHierarchyTheta);
    utils::secureLogValue(_logger, "  Theta (dual hierarchy)", _params.dualHierarchyTheta);

    hdi::utils::secureLog(_logger, "Computing joint probability distribution");  
    _gpgpuHdCompute.init(_params);
    _gpgpuHdCompute.compute(data);    

    if (_params.nLowDimensions == 2) {
      _gpgpu2dSneCompute.init(_gpgpuHdCompute.buffers(), _params);
    } else if (_params.nLowDimensions == 3) {
      _gpgpu3dSneCompute.init(_gpgpuHdCompute.buffers(), _params);
    }

    _iteration = 0;
    _isInit = true;
  }

  void GpgpuTSNE::destr()
  {
    if (!_isInit) {
      return;
    }

    if (_params.nLowDimensions == 2) {
      _gpgpu2dSneCompute.destr();
    } else if (_params.nLowDimensions == 3) {
      _gpgpu3dSneCompute.destr();
    }
    _gpgpuHdCompute.destr();

    _iteration = 0;
    _logger = nullptr;
    _isInit = false;
  }

  void GpgpuTSNE::iterate()
  {
    if (_iteration == _params.momentumSwitchIter) {
      utils::secureLog(_logger, "Switching to final momentum...");
    }

    if (_iteration == _params.removeExaggerationIter) {
      utils::secureLog(_logger, "Removing exaggeration...");
    }

    if (_params.nLowDimensions == 2) {
      _gpgpu2dSneCompute.compute(_iteration, 1.0f);
    } else if (_params.nLowDimensions == 3) {
      _gpgpu3dSneCompute.compute(_iteration, 1.0f);
    }

    ++_iteration;
    
    if (_iteration == _params.iterations - 1) {
      _gpgpuHdCompute.logTimerTotal();
      if (_params.nLowDimensions == 2) {
        _gpgpu2dSneCompute.logTimerAverage();
      } else if (_params.nLowDimensions == 3) {
        _gpgpu3dSneCompute.logTimerAverage();
      }
    }
  }
  
  std::vector<float> GpgpuTSNE::getRawEmbedding() const {
    if (!_isInit) {
      throw std::runtime_error("GpgpuTSNE::getEmbedding() called while GpgpuTSNE was not initialized");
    }

    std::vector<float> embedding(_params.n * _params.nLowDimensions);
    if (_params.nLowDimensions == 2) {
      const auto buffer = _gpgpu2dSneCompute.getEmbedding();
      std::memcpy(embedding.data(), buffer.data(), embedding.size() * sizeof(float));
    } else if (_params.nLowDimensions == 3) {
      const auto buffer = _gpgpu3dSneCompute.getEmbedding();
      std::memcpy(embedding.data(), buffer.data(), embedding.size() * sizeof(float));
    }

    return embedding;
  }

  float GpgpuTSNE::getKLDivergence() {
    if (!_isInit) {
      throw std::runtime_error("GpgpuTSNE::getKLDivergence() called while GpgpuTSNE was not initialized");
    }

    if (_params.nLowDimensions == 2) {
      return GpgpuKldCompute<2>().compute(_gpgpu2dSneCompute.buffers(), _gpgpuHdCompute.buffers(), _params);
    } else if (_params.nLowDimensions == 3) {
      return GpgpuKldCompute<3>().compute(_gpgpu3dSneCompute.buffers(), _gpgpuHdCompute.buffers(), _params);
    } else {
      return std::numeric_limits<float>::infinity();
    }
  }
}