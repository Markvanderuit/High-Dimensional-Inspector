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

#include <cuda_runtime.h>
#include <cuda/helper_math.h>
#include <sstream>
#include <string>

namespace hdi {
  namespace dr {
    namespace bvh {

      /**
       * BVHNode
       * 
       * Simple device-side BVH node definition, padded to 3 * 16 bytes. Stores:
       * - node center of mass
       * - node mass
       * - node bounding sphere radius
       */
      struct BVHNode {
        __device__
        BVHNode(float4 _v, float4 _minb, float4 _maxb)
        : v(_v),
          minb(_minb),
          maxb(_maxb)
        { }

        __device__
        BVHNode()
        : BVHNode(make_float4(0.f, 0.f, 0.f, 0.f), 
                  make_float4(0.f, 0.f, 0.f, 0.f), 
                  make_float4(0.f, 0.f, 0.f, 0.f))
        { }

        float4 v;     // v.xyz = center of mass, v.w = n
        float4 minb;  // minb.xyz = min bounding box, minb.w = diam of bounding box
        float4 maxb;  // maxb.xyz = max bounding box, maxb.w = padding
      };

      /**
       * kernComputeMortonCode
       * 
       * Compute, for each embedding position in pPos, a 30 bit morton code to be stored
       * in pCode. Embedding positions are normalized into [0, 1], so embedding bounds minb
       * and invr = 1 / (maxb - minb) must be provided.
       */
      __global__
      void kernComputeMortonCode(float4 *pPos, uint *pCode, float4 minb, float4 invr, uint n);
      
      /**
       * kernComputeBVHLeaf
       * 
       * Compute and store all leaf nodes for the BVH. There are at least as many leaf nodes
       * to be stored in pNode, as that there are (sorted) embedding positions in pPos. Any
       * extra nodes will be initialized empty and act as padding.
       */
      __global__
      void kernComputeBVHLeaf(float4 *pPos, BVHNode *pNode, uint lvl, uint nPos, uint n);

      
      /**
       * kernComputeBVHNode
       * 
       * Given that BVH leaf nodes are stored in pNode, perform a reduction to store
       * non-leaf nodes in pNode, building a k-ary BVH bottom-up. To-be-constructed levels of the 
       * BVH range from lmax to lmin.
       * 
       * - pNode    ptr to memory in which to read/write nodes.
       * - k        arity of tree (2, 4, 8, horror)
       * - lmax     lowest level to start reducing at
       * - lmin     highest level to end reducing at
       */
      __global__
      void kernComputeBVHNode(BVHNode *pNode, int k, int lmax, int lmin);

      /**
       * kernComputeBVHNode
       * 
       * Given that BVH leaf nodes are stored in pNode, perform a reduction to store
       * non-leaf nodes in pNode, building the BVH bottom-up. To-be-constructed levels of the 
       * BVH range from lmax to lmin.
       */
      __global__
      void kernComputeBVHNode(BVHNode *pNode, int lmax, int lmin);

      /**
       * to_string
       * 
       * Construct a string representation for a BVHNode object.
       */
      inline 
      std::string to_string(const BVHNode& node)
      {
        std::stringstream ss;
        ss << "{ ";
        ss << "com: (" << node.v.x << "," << node.v.y << "," << node.v.z << "), ";
        ss << "n: " << node.v.w << ", ";
        ss << "minb: (" << node.minb.x << "," << node.minb.y << "," << node.minb.z << "), ";
        ss << "maxb: (" << node.maxb.x << "," << node.maxb.y << "," << node.maxb.z << ")";
        ss << " }";
        return ss.str();
      }
    }
  }
}

