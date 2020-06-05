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

#include <random>
#include "hdi/utils/assert_by_exception.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/dimensionality_reduction/sptree.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne.h"

namespace hdi::dr {
  GradientDescentTSNE::GradientDescentTSNE()
  : _isInit(false),
    _logger(nullptr),
    _exaggeration_baseline(1.0),
    _iteration(0)
  { }
  
  GradientDescentTSNE::~GradientDescentTSNE()
  {
    if (_isInit) {
      destr();
    }
  }

  void GradientDescentTSNE::init(const SparseMatrix& P,
                                    Embedding* embeddingPtr,
                                    TsneParameters params)
  {
    utils::secureLog(_logger, "Initializing tSNE...");
    _embeddingPtr = embeddingPtr;
    _params = params;
    _iteration = 0;
    
    // Initialize the high-dimensional probability distribution
    utils::secureLog(_logger, "Computing high-dimensional joint probability distribution...");
    {
      const unsigned n = P.size();
      _P.clear();
      _P.resize(n);

      for (int j = 0; j < n; j++) {
        const auto& p = P[j];

        for (const auto& elem : p) {
          float v = 0.5 * elem.second;

          const auto iter = P[elem.first].find(j);
          if (iter != P[elem.first].end()) {
            v += 0.5 * iter->second;
          }

          // Set Pji and Pij to (Pi|j + Pj|i) / 2
          _P[j][elem.first] = v; 
          _P[elem.first][j] = v; 
        }
      }
    }

    // Initialize the embedding to random positions
    utils::secureLog(_logger, "Initializing embedding...");
    {
      const unsigned n = P.size();
      _embeddingPtr->resize(
        _params._embedding_dimensionality, 
        n, 
        0,
         _params._embedding_dimensionality == 3 ? 1 : 0);

      std::srand(_params._seed);
      for (int i = 0; i < n; ++i) {
        std::vector<double> d(_params._embedding_dimensionality, 0.0);

        double radius;
        do {
          radius = 0.0;
          for (auto& dim : d) {
            dim = 2 * (rand() / ((double) RAND_MAX + 1)) - 1; 
            radius += (dim * dim);
          }
        } while ((radius >= 1.0) || (radius == 0.0));

        radius = sqrt(-2 * log(radius) / radius);
        for (int j = 0; j < _params._embedding_dimensionality; ++j) {
          _embeddingPtr->dataAt(i, j) = d[j] * radius * _params._rngRange;
        }
      }
    }
    
    utils::secureLog(_logger, "Initializing gpu components...");
    switch (_params._embedding_dimensionality) {
      case 2:
        _gpgpu_2d_sne_compute.initialize(_embeddingPtr, _params, _P);
        break;
      case 3:
        _gpgpu_3d_sne_compute.initialize(_embeddingPtr, _params, _P);
        break;
      default:
        break;
    }

    _isInit = true;
  }

  void GradientDescentTSNE::destr()
  {
    switch (_params._embedding_dimensionality) {
      case 2:
        _gpgpu_2d_sne_compute.clean();
        break;
      case 3:
        _gpgpu_3d_sne_compute.clean();
        break;
      default:
        break;
    }
    _embeddingPtr = nullptr;
    _P.clear();
    _iteration = 0;
    _isInit = false;
  }

  void GradientDescentTSNE::iterate(double mult)
  {
    if (_iteration == _params._mom_switching_iter) {
      utils::secureLog(_logger, "Switch to final momentum...");
    }

    if (_iteration == _params._remove_exaggeration_iter) {
      utils::secureLog(_logger, "Remove exaggeration...");
    }

    switch (_params._embedding_dimensionality) {
      case 2:
        _gpgpu_2d_sne_compute.compute(_embeddingPtr,
                                    computeExaggeration(),
                                    _iteration,
                                    mult);
        break;
      case 3:
        _gpgpu_3d_sne_compute.compute(_embeddingPtr,
                                    computeExaggeration(),
                                    _iteration,
                                    mult);
        break;
      default:
        break;
    }
    ++_iteration;
  }

  double GradientDescentTSNE::computeKullbackLeiblerDivergence() const
  {
    const unsigned n = _embeddingPtr->numDataPoints();

    double sum_Q = 0.0;
    #pragma omp parallel for reduction(+: sum_Q)
    for (int j = 0; j < n; ++j) {
      double _sum_Q = 0.0;
      for (int i = j + 1; i < n; ++i) {
        const double euclidean_dist_sq(
          utils::euclideanDistanceSquared<float>(
            _embeddingPtr->getContainer().begin() + j * _params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + (j + 1) * _params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + i * _params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + (i + 1) * _params._embedding_dimensionality
            )
        );
        const double v = 1. / (1. + euclidean_dist_sq);
        _sum_Q += v * 2;
      }
      sum_Q += _sum_Q;
    }

    double kl = 0.0;
    #pragma omp parallel for reduction(+: kl)
    for (int i = 0; i < n; ++i) {
      double _kl = 0.0;
      for (const auto& pij : _P[i]) {
        uint32_t j = pij.first;
        // Calculate Qij
        const double euclidean_dist_sq(
          utils::euclideanDistanceSquared<float>(
            _embeddingPtr->getContainer().begin() + j*_params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + (j + 1)*_params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + i*_params._embedding_dimensionality,
            _embeddingPtr->getContainer().begin() + (i + 1)*_params._embedding_dimensionality
            )
        );
        const double v = 1. / (1. + euclidean_dist_sq);
        double p = pij.second / (2 * n);
        double klc = p * std::log(p / (v / sum_Q));
        _kl += klc;
      }
      kl += _kl;
    }
    return kl;
  }

  double GradientDescentTSNE::computeExaggeration() const
  {
    float exaggeration = _exaggeration_baseline;

    if (_iteration <= _params._remove_exaggeration_iter) {
      exaggeration = _params._exaggeration_factor;
    } else if (_iteration <= (_params._remove_exaggeration_iter + _params._exponential_decay_iter)) {
      double decay = 1.0 - double(_iteration - _params._remove_exaggeration_iter) / _params._exponential_decay_iter;
      exaggeration = _exaggeration_baseline + (_params._exaggeration_factor - _exaggeration_baseline)*decay;
    }

    return exaggeration;
  }
}