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


#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda/helper_math.h>
#include <cub/cub.cuh>
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"

namespace hdi {
  namespace dr {
    namespace bvh {
      __device__
      inline
      uint expandBits(uint i)
      {
        i = (i * 0x00010001u) & 0xFF0000FFu;
        i = (i * 0x00000101u) & 0x0F00F00Fu;
        i = (i * 0x00000011u) & 0xC30C30C3u;
        i = (i * 0x00000005u) & 0x49249249u;
        return i;
      }
      
      __device__
      inline
      uint mortonCode(float4 f)
      {
        f = clamp(f * 1024.f, 0.f, 1023.f);
        uint x = expandBits(static_cast<uint>(f.x));
        uint y = expandBits(static_cast<uint>(f.y));
        uint z = expandBits(static_cast<uint>(f.z));
        return x * 4u + y * 2u + z;
      }

      __global__
      void kernConstrMorton(BVHLayout layout, 
        float4 minb, float4 invr, float4 *pEmb, uint *pMorton)
      {
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= layout.nPos) {
          return;
        }
      
        // Normalize position
        const float4 pos = (pEmb[gid] - minb) * invr;

        // Store morton code
        pMorton[gid] = mortonCode(pos);
      }

      __global__
      void kernConstrLeaves(
        BVHLayout layout, 
        float4 *pPos, 
        float4 *pNode, 
        uint *pMass, 
        uint *pIdx, 
        float4 *pMinB, 
        float4 *pMaxB)
      {
        const uint globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (globalIdx >= layout.nLeaves) {
          return;
        }
        
        // Initial leaf values
        float4 center = make_float4(0);
        float4 minb = make_float4(0);
        float4 maxb = make_float4(0);
        uint mass = 0u;
        
        // Determine nr. of positions under leaf
        const uint posIdx = globalIdx * layout.kLeaf;
        int borderValue = static_cast<int>(layout.nPos) - static_cast<int>(posIdx);
        if (borderValue > 0) {
          // Add values of first node
          mass = min(layout.kLeaf, static_cast<uint>(borderValue));
          center = pPos[posIdx];
          minb = center;
          maxb = center;
          
          // Add values of rest of nodes
          for (uint i = 1; i < mass; ++i) {
            float4 pos = pPos[posIdx + i];
            center += pos;
            minb = fminf(minb, pos);
            maxb = fmaxf(maxb, pos);
          }

          // Average
          center /= static_cast<float>(max(mass, 1u));
          float4 diamsv = maxb - minb;
          center.w = dot(diamsv, diamsv); // Compute squared diameter
        }

        // Store leaf values
        uint leafAddr = layout.nNodes + globalIdx;
        pNode[leafAddr] = center;
        pMass[leafAddr] = mass;
        pMinB[leafAddr] = minb;
        pMaxB[leafAddr] = maxb;
        pIdx[globalIdx] = posIdx;
      }

      struct AONode {
        float4 node;
        float4 minb;
        float4 maxb;
        uint mass;
      };

      struct AOReductor {
        __device__
        AONode operator()(const AONode &l, const AONode &r) {
          // Escape empty nodes
          if (l.mass == 0) {
            return r;
          } else if (r.mass == 0) {
            return l;
          }
          
          // Reduce node from two non-empty nodes
          AONode node;
          node.mass = l.mass + r.mass;
          node.node = (static_cast<float>(l.mass) * l.node + static_cast<float>(r.mass) * r.node) / static_cast<float>(node.mass);
          node.minb = fminf(l.minb, r.minb);
          node.maxb = fmaxf(l.maxb, r.maxb);

          float4 diamv = node.maxb - node.minb;
          node.node.w = dot(diamv, diamv);

          return node;
        }
      };

      template <uint KNode>
      __global__
      void kernConstrNodes(
        BVHLayout layout,
        uint level, uint lNodes, uint lOffset,
        float4 *pNode, uint *pMass, float4 *pMinB, float4 *pMaxB)
      {
        typedef cub::WarpReduce<AONode, KNode> WarpReduce;
        
        // Allocate WarpReduce shared memory for 1 warp
        __shared__ typename WarpReduce::TempStorage temp_storage;

        const uint globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (globalIdx >= lNodes) {
          return;
        }

        // Obtain input item per thread
        const uint logk = static_cast<uint>(log2f(layout.kNode));
        uint posIdx = lOffset + globalIdx;
        AONode node = {
          pNode[posIdx],
          pMinB[posIdx],
          pMaxB[posIdx],
          pMass[posIdx]
        };
        
        // Perform reduction
        AONode aggregate = WarpReduce(temp_storage).Reduce(node, AOReductor());

        // Store in lower level
        posIdx = lOffset - (1u << (logk * (level - 1))) + globalIdx / 8;
        if (globalIdx % 8 == 0) {
          pNode[posIdx] = aggregate.node;
          pMinB[posIdx] = aggregate.minb;
          pMaxB[posIdx] = aggregate.maxb;
          pMass[posIdx] = aggregate.mass;
        }
      }

      template __global__
      void kernConstrNodes<2>(BVHLayout, uint, uint, uint, float4*, uint*, float4*, float4*);
      template __global__
      void kernConstrNodes<4>(BVHLayout, uint, uint, uint, float4*, uint*, float4*, float4*);
      template __global__
      void kernConstrNodes<8>(BVHLayout, uint, uint, uint, float4*, uint*, float4*, float4*);
      template __global__
      void kernConstrNodes<16>(BVHLayout, uint, uint, uint, float4*, uint*, float4*, float4*);
    }
  }
}