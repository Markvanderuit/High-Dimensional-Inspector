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

#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>
#include <string>

typedef unsigned uint;

namespace hdi {
  namespace dr {
    namespace bvh {
      template <typename Int> 
      inline
      Int ceilDiv(Int n, Int div) {
        return (n + div - 1) / div;
      }

      /**
       * Layout specifics of implicit Bounding Volume Hierarchy (BVH) structure. The
       * structure supports different fan-out (arity) parameters for leaf and 
       * non-leaf nodes.
       * 
       * @author Mark van de Ruit
       */
      class BVHLayout {
      public:
        uint kNode;   // Fan-out of non-leaf nodes
        uint nPos;    // Nr of embedding points stored in tree
        uint nLvls;   // Nr of tree levels
        uint nLeaves; // Nr of leaf nodes
        uint nNodes;  // Nr of nodes total

        /**
         * Constructor computes specific layout based on arity k
         * and number of contained embedding points nPos.
         */
        BVHLayout(uint kNode, uint nPos)
        : kNode(kNode), 
          nPos(nPos),
          nLeaves(nPos)
        {
          const uint logk = static_cast<uint>(std::log2(kNode));

          nLvls = static_cast<uint>(std::ceil(std::log2(nLeaves) / logk)) + 1;

          // Raise to nearest convenient nr of leaves for this fanout
          nLeaves = 1u << (logk * (nLvls - 1));

          // Compute nr of nodes iteratively (a bit more work because kNode >= 2)
          nNodes = 0u;
          for (uint i = 0u; i < nLvls; i++) {
            nNodes |= 1u << (logk * i); 
          }
        }

        /**
         * Default constr.
         */
        BVHLayout() = default;
      };

      inline
      std::string to_string(const BVHLayout& layout)
      {
        std::stringstream ss;
        ss << "BVHLayout {\n";
        ss << "  kNode    " << layout.kNode << '\n';
        ss << "  nPos     " << layout.nPos << '\n';
        ss << "  nLvls    " << layout.nLvls << '\n';
        ss << "  nLeaves  " << layout.nLeaves << '\n';
        ss << "  nNodes   " << layout.nNodes << '\n';
        ss << "}";
        return ss.str();
      }
    }
  }
}