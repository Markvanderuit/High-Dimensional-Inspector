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

#include <limits>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace hdi {
  namespace dr {
    using uint = unsigned;

    namespace detail {
      constexpr unsigned std430_align(unsigned D) {
        return D == 4 ? 16
             : D == 3 ? 16
             : D == 2 ? 8
             : 4;
      }
    }

    template <unsigned D, typename genType>
    struct alignas(detail::std430_align(D)) AlignedVec : public glm::vec<D, genType> {
      using glm::vec<D, genType>::vec;

      AlignedVec<D, genType>(glm::vec<D, genType> other)
      : glm::vec<D, genType>(other)
      { }
    };

    template< unsigned D, typename genType >
    genType max(AlignedVec<D, genType> x) {
      genType t = x[0];
      for (uint i = 1; i < D; i++) {
        t = glm::max(t, x[i]);
      }
      return t;
    }

    template< unsigned D, typename genType >
    genType min(AlignedVec<D, genType> x) {
      genType t = x[0];
      for (uint i = 1; i < D; i++) {
        t = glm::min(t, x[i]);
      }
      return t;
    }

    template< unsigned D, typename genType >
    AlignedVec<D, genType> max(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
      return glm::max(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
    }

    template< unsigned D, typename genType >
    AlignedVec<D, genType> min(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
      return glm::min(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
    }

    template< unsigned D, typename genType >
    std::string to_string(AlignedVec<D, genType> x) {
      return glm::to_string(static_cast<glm::vec<D, genType>>(x));
    }

    template< unsigned D, typename genType >
    genType product(AlignedVec<D, genType> x) {
      genType t = x[0];
      for (uint i = 1; i < D; i++) {
        t *= x[i];
      }
      return t;
    }

    template< unsigned D, typename genType >
    genType dot(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
      return glm::dot(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
    }

    // Rounded up division of n by div
    template <typename genType> 
    inline
    genType ceilDiv(genType n, genType div) {
      return (n + div - 1) / div;
    }

    /**
     * Simple bounding box type for D dimensions.
     */
    template <unsigned D>
    class AlignedBounds {
    private:
      using vec = AlignedVec<D, float>;

    public:
      vec min, max;

      vec range() const {
        return max - min;
      }

      vec center() const {
        return 0.5f * (max + min);
      }
    };
  }
}