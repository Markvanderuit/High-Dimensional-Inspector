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
      // inline
      uint expandBits(uint i)
      {
        i = (i * 0x00010001u) & 0xFF0000FFu;
        i = (i * 0x00000101u) & 0x0F00F00Fu;
        i = (i * 0x00000011u) & 0xC30C30C3u;
        i = (i * 0x00000005u) & 0x49249249u;
        return i;
      }
      
      __device__
      // inline
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
        float4 minb, float4 invr, 
        float4 *pEmb, uint *pMorton)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.nPos) {
          return;
        }
      
        // Normalize position to [0, 1]
        const float4 pos = (pEmb[i] - minb) * invr;

        // Compute and store morton code
        pMorton[i] = mortonCode(pos);
      }

      __global__
      void kernConstrPos(
        BVHLayout layout,
        uint *idx, float4 *posIn, float4 *posOut)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.nPos) {
          return;
        }
        
        posOut[i] = posIn[idx[i]];
      }

      __device__
      uint findSplit(uint *pMorton, uint level, uint levels, uint first, uint last)
      {
        uint firstCode = pMorton[first];
        uint lastCode = pMorton[last];
        uint commonPrefix = __clz(firstCode ^ lastCode);
        
        // Initial guess for split position
        uint split = first;
        uint step = last - first;

        // Perform a binary search to find the split position
        do {
          step = (step + 1) >> 1; // Decrease step size
          uint _split = split + step; // Possible new split position

          if (_split < last) {
            uint splitCode = pMorton[_split];
            uint splitPrefix = __clz(firstCode ^ splitCode);

            // Accept newly proposed split for this iteration
            if (splitPrefix > commonPrefix) {
              split = _split;
            }
          }
        } while (step > 1);

        // Split at half point if a really bad split is found
        // if (levels - level < 5) {
        //   float relation = static_cast<float>(last - split) / static_cast<float>(last - first);
        //   if (last - first > 1 && relation < 0.8f || relation > 0.8f) {
        //   // if (last - split > 2 && static_cast<float>(last - split) / static_cast<float>(last - first) <= 0.33) {
        //     split = first + (last - first) / 2;
        //   }
        // }
        
        // if ((last - split < 2 || split - first < 2) && last - first > 0) {
        // }

        return split;
      }
      
      __global__
      void kernConstrSubdiv(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        uint *pMorton, uint *pIdx, float4 *pPos, float4 *pNode, float4 *pMinB, float4 *pDiam)
      {
        const uint i = begin + blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= end) {
          return;
        }
        
        // Read in parent range into shared memory
        __shared__ uint parentData[256];
        if (threadIdx.x % 2 == 0) {
          parentData[threadIdx.x] = pIdx[(i >> 1) - 1];
        } else {
          parentData[threadIdx.x] = static_cast<uint>(pNode[(i >> 1) - 1].w);
        }
        __syncthreads();
        // uint *parentPtr = (threadIdx.x % 2 == 0) ? pIdx : pMass;
        // parentData[threadIdx.x] = static_cast<uint>(parentPtr[(i >> 1) - 1]);
        
        const uint t = threadIdx.x - (threadIdx.x % 2);
        const uint parentIdx = parentData[t];
        const uint parentMass = parentData[t + 1];
        uint idx = 0u;
        float4 node = make_float4(0);
        
        if (parentMass > 1) {
          // Find split point and subdivide
          uint rangeBegin = parentIdx;
          uint rangeEnd = rangeBegin + parentMass - 1;
          uint split = findSplit(pMorton, level, levels, rangeBegin, rangeEnd);
          
          if (threadIdx.x % 2 == 0) {
            // Assign left tree values
            idx = rangeBegin;
            node.w = 1 + split - rangeBegin;
          } else {
            // Assign right tree values
            idx = split + 1;
            node.w = rangeEnd - split;
          }
        }

        // Compute leaf node data if necessary
        if (node.w == 1 || (node.w > 0 && level == levels - 1)) {
          float4 pos = pPos[idx];
          float4 minb = pos;
          float4 maxb = pos;
          for (uint _idx = idx + 1; _idx < idx + static_cast<uint>(node.w); _idx++) {
            float4 _pos = pPos[_idx];
            minb = fminf(minb, _pos);
            maxb = fmaxf(maxb, _pos);
            pos += _pos;
          }
          pos /= static_cast<float>(node.w);

          node = make_float4(pos.x, pos.y, pos.z, node.w);
          pMinB[i - 1] = minb;
          pDiam[i - 1] = maxb - minb;
        }

        pIdx[i - 1] = idx;
        pNode[i - 1] = node;
      }
      
      struct ReducableNode {
        float4 node;
        float4 minb;
        float4 diam;

        __device__
        ReducableNode()
        { }

        __device__
        ReducableNode(float4 node, float4 minb, float4 diam)
        : node(node), minb(minb), diam(diam)
        { }

        __device__
        ReducableNode(const ReducableNode &l, const ReducableNode &r)
        : minb(fminf(l.minb, r.minb)),
          diam(fmaxf(l.diam + l.minb, r.diam + r.minb) - minb)
        { 
          float mass = l.node.w + r.node.w;
          node = (l.node.w * l.node + r.node.w * r.node) / mass;
          node.w = mass;
        }
      };

      __device__
      ReducableNode reduce(ReducableNode l, ReducableNode r)
      {
        if (r.node.w != 0u) {
          return ReducableNode(l, r);
        } else {
          return l;
        }
      } 

      __device__
      ReducableNode read(uint i, float4 *pNode, float4 *pMinB, float4 *pDiam)
      {
        const float4 node = pNode[i];
        if (node.w > 0.f) {
          return ReducableNode(node, pMinB[i], pDiam[i]);
        } else {
          return ReducableNode(make_float4(0), make_float4(0), make_float4(0));
        }
      }

      __device__
      void write(uint i, ReducableNode &node, float4 *pNode, float4 *pMinB, float4 *pDiam)
      {
        pNode[i] = node.node;
        pMinB[i] = node.minb;
        pDiam[i] = node.diam;
      }

      __global__
      void kernConstrDataReduce(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        uint *pIdx, float4 *pPos, float4 *pNode, float4 *pMinB, float4 *pDiam)
      {
        __shared__ ReducableNode reductionNodes[32];
        const uint halfBlockDim = blockDim.x / 2;

        const uint t = threadIdx.x;
        const uint halft = t / 2;
        uint i = begin + (blockIdx.x * blockDim.x + threadIdx.x);
        if (i >= end) {
          return;
        }
        
        ReducableNode node = read(i - 1, pNode, pMinB, pDiam);
        i >>= 1;
        
        if (t % 2 == 1) {
          reductionNodes[halft] = node;
        }
        __syncthreads();
        if (t % 2 == 0) {
          node = reduce(node, reductionNodes[halft]);
          if (node.node.w == 0.f) {
            reductionNodes[halft] = read(i - 1, pNode, pMinB, pDiam);
          } else {
            reductionNodes[halft] = node;
            write(i - 1, node, pNode, pMinB, pDiam);
          }
        }
        if (--level == 0) {
          return;
        }

        uint mod = 4;

        for (uint j = halfBlockDim / 2; j > 1; j /= 2) {
          __syncthreads();
          i >>= 1;
          
          if (t % mod == 0) {
            node = reduce(reductionNodes[halft], reductionNodes[halft + (mod / 4)]);
            if (node.node.w == 0.f) {
              reductionNodes[halft] = read(i - 1, pNode, pMinB, pDiam);
            } else {
              reductionNodes[halft] = node;
              write(i - 1, node, pNode, pMinB, pDiam);
            }
          }
          
          mod *= 2;
          if (--level == 0) {
            return;
          }
        }
        
        // Write out top layer
        __syncthreads();
        i >>= 1;
        if (t == 0) {
          node = reduce(reductionNodes[0], reductionNodes[halfBlockDim / 2]);
          if (node.node.w > 0.f) {
            write(i - 1, node, pNode, pMinB, pDiam);
          }
        }
      }
      
      __global__
      void kernConstrCleanup(
        uint n, float4 *pNode, float4 *pMinB, float4 *pDiam)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) {
          return;
        }
        
        if (pNode[i].w == 0) {
          pMinB[i] = make_float4(0);
          pDiam[i] = make_float4(0);
        }
      }
    }
  }
}