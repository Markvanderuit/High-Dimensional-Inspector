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

#include <vector>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/type_aligned.hpp>
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"

namespace hdi {
  namespace dr {
    typedef unsigned uint;
    typedef data::Embedding<float> Embedding;
    typedef std::vector<data::MapMemEff<uint32_t, float>> SparseMatrix;
    
    // Rounded up division of T over T
    template <typename genType> 
    inline
    genType ceilDiv(genType n, genType div) {
      return (n + div - 1) / div;
    }

    // Product of a glm::vec<D, T>'s components
    template <typename genType, uint D>
    inline
    genType product(glm::vec<D, genType, glm::aligned_highp> v) {
      genType t = v[0];
      for (uint i = 1; i < D; i++) {
        t *= v[i];
      }
      return t;
    }

    /**
     * Simple bounding box type for D dimensions.
     */
    template <unsigned D>
    class Bounds {
    private:
      typedef glm::vec<D, float, glm::aligned_highp> vec;

    public:
      vec min, max;

      vec range() const {
        return max - min;
      }

      vec center() const {
        return 0.5f * (max + min);
      }
    };

    /**
     * Linearized version of probability matrix
     */
    struct LinearProbabilityMatrix {
      std::vector<uint32_t> neighbours;
      std::vector<float> probabilities;
      std::vector<int> indices;
    };

    /**
     * Compute linearized version of sparse matrix
     */
    inline LinearProbabilityMatrix linearizeProbabilityMatrix(const Embedding *embeddingPtr, 
                                                              const SparseMatrix &P) {
      LinearProbabilityMatrix LP;
      LP.indices.reserve(2 * embeddingPtr->numDataPoints());
      LP.neighbours.reserve(128 * embeddingPtr->numDataPoints());
      LP.probabilities.reserve(128 * embeddingPtr->numDataPoints());

      for (size_t i = 0; i < embeddingPtr->numDataPoints(); i++) {
        LP.indices.push_back(static_cast<int>(LP.neighbours.size()));
        for (const auto& p_ij : P[i]) {
          LP.neighbours.push_back(p_ij.first);
          LP.probabilities.push_back(p_ij.second);
        }
        LP.indices.push_back(static_cast<int>(P[i].size()));
      }

      return LP;
    }
  }
}