/*
 *
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
 *
 */

#include "hdi/dimensionality_reduction/gradient_descent_tsne_3d.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "sptree.h"
#include <random>

// uh-oh
#pragma warning( push )
#pragma warning( disable : 4267)
#pragma warning( disable : 4291)
#pragma warning( disable : 4996)
#pragma warning( disable : 4018)
#pragma warning( disable : 4244)
#include "flann/flann.h"
#pragma warning( pop )

namespace hdi::dr {
  GradientDescentTSNE3D::GradientDescentTSNE3D()
  : _initialized(false),
    _logger(nullptr),
    _exaggeration_baseline(1.0) { }

  GradientDescentTSNE3D::~GradientDescentTSNE3D()
  { }

  void GradientDescentTSNE3D::initialize(const sparse_scalar_matrix_t& probabilities,
                                         data::Embedding<scalar_t>* embedding, 
                                         TsneParameters params) {
    utils::secureLog(_logger, "Initializing 3d-tSNE...");

    _params = params;
    _embedding = embedding;
    unsigned int n = probabilities.size();
    
    utils::secureLog(_logger, "Computing high-dimensional joint probability distribution...");

    // Properly size P
    _P.clear();
    _P.resize(n);

    // Fill P following formula
    for (int j = 0; j < n; j++) {
      for (const auto& elem : probabilities[j]) {
        scalar_t v0 = elem.second, v1 = 0.0;
        const auto iter = probabilities[elem.first].find(j);
        if (iter != probabilities[elem.first].end()) {
          v1 = iter->second;
        }

        // Set Pji and Pij to (Pi|j + Pj|i) / 2
        _P[j][elem.first] = static_cast<scalar_t>((v0 + v1) * 0.5); 
        _P[elem.first][j] = static_cast<scalar_t>((v0 + v1) * 0.5); 
      }
    }

    // Properly size embedding, add 1 float padding for vec4 on gpu
    _embedding->resize(_params._embedding_dimensionality, n, 0, 1);
  
    utils::secureLog(_logger, "Initializing the embedding...");

    // Initialize embedding to random positions
    std::srand(_params._seed);
    for (int i = 0; i < _embedding->numDataPoints(); ++i) {
      std::vector<double> d(_embedding->numDimensions(), 0.0);
      double radius;
      do {
        radius = 0.0;
        for (auto& dim : d) {
          dim =  2 * (rand() / ((double)RAND_MAX + 1)) - 1; 
          radius += (dim * dim);
        }
      } while ((radius >= 1.0) || (radius == 0.0));

      radius = sqrt(-2 * log(radius) / radius);
      for (int j = 0; j < _embedding->numDimensions(); ++j) {
        _embedding->dataAt(i, j) = d[j] * radius * _params._rngRange;
      }
    }
    
    // Initialize underlying tsne implementation
    _gpgpu_3d_sne_compute.initialize(_embedding, _params, _P);
    _iteration = 0;
    _initialized = true;
    utils::secureLog(_logger, "Initialization complete!");
  }

  void GradientDescentTSNE3D::iterate(double mult) {
    if (!_initialized) {
      throw std::logic_error("Cannot compute a gradient descent iteration on unitialized data");
    }
    if (_iteration == _params._mom_switching_iter) {
      utils::secureLog(_logger, "Switch to final momentum...");
    }
    if (_iteration == _params._remove_exaggeration_iter) {
      utils::secureLog(_logger, "Remove exaggeration...");
    }

    // Forward to underlying tsne implementation
    _gpgpu_3d_sne_compute.compute(_embedding, 
                                  computeExaggeration(), 
                                  _iteration, 
                                  mult);
    ++_iteration;
  }

  double GradientDescentTSNE3D::computeExaggeration() {
    scalar_t exaggeration = _exaggeration_baseline;

    if (_iteration <= _params._remove_exaggeration_iter) {
      exaggeration = _params._exaggeration_factor;
    } else if (_iteration <= (_params._remove_exaggeration_iter + _params._exponential_decay_iter)) {
      double decay = 1.0 - double(_iteration - _params._remove_exaggeration_iter) / _params._exponential_decay_iter;
      exaggeration = _exaggeration_baseline + (_params._exaggeration_factor - _exaggeration_baseline)*decay;
    }

    return exaggeration;
  }

  double GradientDescentTSNE3D::computeKullbackLeiblerDivergence() {
    const int n = _embedding->numDataPoints();

    double sum_Q = 0;
    for (int j = 0; j < n; ++j) {
      for (int i = j + 1; i < n; ++i) {
        const double euclidean_dist_sq(
          utils::euclideanDistanceSquared<float>(
            _embedding->getContainer().begin() + j * _params._embedding_dimensionality,
            _embedding->getContainer().begin() + (j + 1) * _params._embedding_dimensionality,
            _embedding->getContainer().begin() + i * _params._embedding_dimensionality,
            _embedding->getContainer().begin() + (i + 1) * _params._embedding_dimensionality
            )
        );
        const double v = 1. / (1. + euclidean_dist_sq);
        sum_Q += v * 2;
      }
    }

    double kl = 0;
    for (int i = 0; i < n; ++i) {
      for (const auto& pij : _P[i]) {
        uint32_t j = pij.first;

        // Calculate Qij
        const double euclidean_dist_sq(
          utils::euclideanDistanceSquared<float>(
            _embedding->getContainer().begin() + j*_params._embedding_dimensionality,
            _embedding->getContainer().begin() + (j + 1)*_params._embedding_dimensionality,
            _embedding->getContainer().begin() + i*_params._embedding_dimensionality,
            _embedding->getContainer().begin() + (i + 1)*_params._embedding_dimensionality
            )
        );
        const double v = 1. / (1. + euclidean_dist_sq);

        double p = pij.second / (2 * n);
        float klc = p * std::log(p / (v / sum_Q));
        kl += klc;
      }
    }
    return kl;
  }
}