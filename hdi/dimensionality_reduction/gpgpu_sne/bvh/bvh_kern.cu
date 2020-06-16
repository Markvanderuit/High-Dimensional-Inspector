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
      // 2 0-bit interleaving for 10-bit expansion
      __device__
      uint expandBits10(uint i)
      {
        i = (i * 0x00010001u) & 0xFF0000FFu;
        i = (i * 0x00000101u) & 0x0F00F00Fu;
        i = (i * 0x00000011u) & 0xC30C30C3u;
        i = (i * 0x00000005u) & 0x49249249u;
        return i;
      }

      // 1 0-bit interleaving for 15-bit expansion
      __device__
      uint expandBits15(uint i) 
      {
        i = (i | (i << 8u)) & 0x00FF00FFu;
        i = (i | (i << 4u)) & 0x0F0F0F0Fu;
        i = (i | (i << 2u)) & 0x33333333u;
        i = (i | (i << 1u)) & 0x55555555u;
        return i;
      }

      template <unsigned D>
      __device__
      uint mortonCode(vec<D> f);

      template <>
      __device__
      uint mortonCode<2>(vec<2> f)
      {
        f = clamp(f * 32768.f, 0.f, 32767.f);

        const uint x = expandBits15(static_cast<uint>(f.x));
        const uint y = expandBits15(static_cast<uint>(f.y));

        return x | (y << 1);
      }

      template <>
      __device__
      uint mortonCode<3>(vec<3> f)
      {
        f = clamp(f * 1024.f, 0.f, 1023.f);
        
        const uint x = expandBits10(static_cast<uint>(f.x));
        const uint y = expandBits10(static_cast<uint>(f.y));
        const uint z = expandBits10(static_cast<uint>(f.z));

        return x * 4u + y * 2u + z;
      }

      template <unsigned D>
      __global__
      void kernConstrMorton(
        BVHLayout layout, 
        BVHBounds<D> *pBounds, vec<D> *pEmb, uint *pMorton)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.nPos) {
          return;
        }
      
        // Normalize position to [0, 1]
        const vec<D> pos = (pEmb[i] - pBounds->min) * pBounds->invrange;

        // Compute and store morton code
        pMorton[i] = mortonCode<D>(pos);
      }

      // Explicit template instantiations
      template __global__
      void kernConstrMorton<2>(BVHLayout, BVHBounds<2> *, vec<2> *, uint *);
      template __global__
      void kernConstrMorton<3>(BVHLayout, BVHBounds<3> *, vec<3> *, uint *);

      template <unsigned D>
      __global__
      void kernConstrPos(
        BVHLayout layout,
        uint *idx, vec<D> *posIn, vec<D> *posOut)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.nPos) {
          return;
        }
        
        posOut[i] = posIn[idx[i]];
      }

      // Explicit template instantiations
      template __global__
      void kernConstrPos<2>(BVHLayout, uint*, vec<2>*, vec<2>*);
      template __global__
      void kernConstrPos<3>(BVHLayout, uint*, vec<3>*, vec<3>*);

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
        //   if (last - first > 1 && relation < 0.33f || relation > 0.66f) {
        //   // if (last - split > 2 && static_cast<float>(last - split) / static_cast<float>(last - first) <= 0.33) {
        //     split = first + (last - first) / 2;
        //   }
        // }
        
        // if ((last - split < 2 || split - first < 2) && last - first > 0) {
        // }

        return split;
      }
      
      constexpr uint minMass = 8;

      template <unsigned D>
      __global__
      void kernConstrSubdiv(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        uint *pMorton, vec<D> *pPos, float4 *pNode, vec<D> *pMinB, float4 *pDiam)
      {
        const uint i = begin + blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= end) {
          return;
        }
        
        // Read in parent range into shared memory
        __shared__ uint parentData[256];
        if (threadIdx.x % 2 == 0) {
          parentData[threadIdx.x] = static_cast<uint>(pDiam[(i >> 1) - 1].w);
        } else {
          parentData[threadIdx.x] = static_cast<uint>(pNode[(i >> 1) - 1].w);
        }
        __syncthreads();
        
        const uint t = threadIdx.x - (threadIdx.x % 2);
        const uint parentIdx = parentData[t];
        const uint parentMass = parentData[t + 1];
        // uint idx = 0u;
        float4 node = make_float4(0);
        float4 diam = make_float4(0);
        
        if (parentMass > minMass) {
          // Find split point and subdivide
          uint rangeBegin = parentIdx;
          uint rangeEnd = rangeBegin + parentMass - 1;
          uint split = findSplit(pMorton, level, levels, rangeBegin, rangeEnd);
          
          if (threadIdx.x % 2 == 0) {
            // Assign left tree values
            // idx = rangeBegin;
            diam.w = rangeBegin;
            node.w = 1 + split - rangeBegin;
          } else {
            // Assign right tree values
            // idx = split + 1;
            diam.w = split + 1;
            node.w = rangeEnd - split;
          }
        }

        // Compute leaf node data if necessary
        if ((node.w > 0 && node.w <= minMass) || (node.w > 0 && level == levels - 1)) {
          vec<D> pos = pPos[uint(diam.w)];
          vec<D> minb = pos;
          vec<D> maxb = pos;
          for (uint _idx = uint(diam.w) + 1; _idx < uint(diam.w) + uint(node.w); _idx++) {
            vec<D> _pos = pPos[_idx];
            minb = fminf(minb, _pos);
            maxb = fmaxf(maxb, _pos);
            pos += _pos;
          }
          pos /= static_cast<float>(node.w);
          vec<D> _diam = maxb - minb;
          
          if constexpr (D == 2) {
            node = make_float4(pos.x, pos.y, 0.f, node.w);
            diam = make_float4(_diam.x, _diam.y, 0.f, diam.w);
          } else if constexpr (D == 3) {
            node = make_float4(pos.x, pos.y, pos.z, node.w);
            diam = make_float4(_diam.x, _diam.y, _diam.z, diam.w);
          }

          pMinB[i - 1] = minb;
        }

        pNode[i - 1] = node;
        pDiam[i - 1] = diam;
      }
      
      // Explicit template instantiations
      template  __global__
      void kernConstrSubdiv<2>(
        BVHLayout, uint, uint, uint, uint, uint*, vec<2>*, float4*, vec<2>*, float4*);
      template  __global__
      void kernConstrSubdiv<3>(
        BVHLayout, uint, uint, uint, uint, uint*, vec<3>*, float4*, vec<3>*, float4*);
      
//       template <unsigned D>
//       __global__
//       void kernConstrLeaf(
//         BVHLayout layout,
//         uint begin, uint end,
//         uint *pIdx, vec<D> *pPos, float4 *pNode, vec<D> *pMinB, vec<D> *pDiam)
//       {
//         const uint i = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i >= end) {
//           return;
//         }

// /*         float4 node = pNode[i];
        
//         // Skip empty nodes
//         if (node.w == 0) {
//           return;
//         }
        
//         // Skip non-leaf nodes
//         if (node.w > 1 && i < begin) {
//           return;
//         }
        
//         uint idx = pIdx[i];
//         vec<D> pos = pPos[idx];
//         vec<D> minb = pos;
//         vec<D> maxb = pos;

//         for (uint _idx = idx + 1; _idx < idx + static_cast<uint>(node.w); _idx++) {
//           vec<D> _pos = pPos[_idx];
//           minb = fminf(minb, _pos);
//           maxb = fmaxf(maxb, _pos);
//           pos += _pos;
//         }
//         pos /= static_cast<float>(node.w);
//         if constexpr (D == 2) {
//           node = make_float4(pos.x, pos.y, 0.f, node.w);
//         } else if constexpr (D == 3) {
//           node = make_float4(pos.x, pos.y, pos.z, node.w);
//         }

//         pMinB[i] = minb;
//         pDiam[i] = maxb - minb;
//         pNode[i] = node; */

//         // // QCompute leaf node data if necessary
//         // if (node.w == 1 || (node.w > 0 && level == levels - 1)) {
//         //   vec<D> pos = pPos[idx];
//         //   vec<D> minb = pos;
//         //   vec<D> maxb = pos;
//         //   for (uint _idx = idx + 1; _idx < idx + static_cast<uint>(node.w); _idx++) {
//         //     vec<D> _pos = pPos[_idx];
//         //     minb = fminf(minb, _pos);
//         //     maxb = fmaxf(maxb, _pos);
//         //     pos += _pos;
//         //   }
//         //   pos /= static_cast<float>(node.w);
          
//         //   if constexpr (D == 2) {
//         //     node = make_float4(pos.x, pos.y, 0.f, node.w);
//         //   } else if constexpr (D == 3) {
//         //     node = make_float4(pos.x, pos.y, pos.z, node.w);
//         //   }
//         //   pMinB[i - 1] = minb;
//         //   pDiam[i - 1] = maxb - minb;
//         // }
//       }

      // Explicit template instantiations
      /* template __global__
      void kernConstrLeaf<2>(
        BVHLayout, uint, uint, uint*, vec<2>*, float4*, vec<2>*, vec<2>*);
      template __global__
      void kernConstrLeaf<3>(
        BVHLayout, uint, uint, uint*, vec<3>*, float4*, vec<3>*, vec<3>*); */

      template <unsigned D>
      struct ReducableNode {
        float4 node;
        vec<D> minb;
        float4 diam;

        __device__
        ReducableNode()
        { }

        __device__
        ReducableNode(float4 node, vec<D> minb, float4 diam)
        : node(node), minb(minb), diam(diam)
        { }

        __device__
        ReducableNode(const ReducableNode &l, const ReducableNode &r)
        : minb(fminf(l.minb, r.minb))
        { 
          if constexpr (D == 2) {
            vec<D> _diam = fmaxf(make_float2(l.diam.x, l.diam.y) + l.minb,
                                 make_float2(r.diam.x, r.diam.y) + r.minb) - minb;
            diam = make_float4(_diam.x, _diam.y,  0.f, fminf(l.diam.w, r.diam.w));
          } else if constexpr (D == 3) {
            vec<D> _diam = fmaxf(make_float4(l.diam.x, l.diam.y, l.diam.z, 0) + l.minb,
                                 make_float4(r.diam.x, r.diam.y, r.diam.z, 0) + r.minb) - minb;
            diam = make_float4(_diam.x, _diam.y, _diam.z, fminf(l.diam.w, r.diam.w));
          }

          float mass = l.node.w + r.node.w;
          node = (l.node.w * l.node + r.node.w * r.node) / mass;
          node.w = mass;
        }
      };

      template <unsigned D>
      __device__
      ReducableNode<D> reduce(ReducableNode<D> l, ReducableNode<D> r)
      {
        if (r.node.w != 0u) {
          return ReducableNode<D>(l, r);
        } else {
          return l;
        }
      } 
      
      template <unsigned D>
      __device__
      ReducableNode<D> read(uint i, float4 *pNode, vec<D> *pMinB, float4 *pDiam)
      {
        const float4 node = pNode[i];
        if (node.w > 0.f) {
          return ReducableNode<D>(node, pMinB[i], pDiam[i]);
        } else {
          if constexpr (D == 2) {
            return ReducableNode<D>(make_float4(0), make_float2(0), make_float4(0));
          } else if constexpr (D == 3) {
            return ReducableNode<D>(make_float4(0), make_float4(0), make_float4(0));
          }
          return ReducableNode<D>();
        }
      }

      template <unsigned D>
      __device__
      void write(uint i, ReducableNode<D> &node, float4 *pNode, vec<D> *pMinB, float4 *pDiam)
      {
        pNode[i] = node.node;
        pMinB[i] = node.minb;
        pDiam[i] = node.diam;
      }

      template <unsigned D>
      __global__
      void kernConstrBbox(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        vec<D> *pPos, float4 *pNode, vec<D> *pMinB, float4 *pDiam)
      {
        __shared__ ReducableNode<D> reductionNodes[32];
        const uint halfBlockDim = blockDim.x / 2;

        const uint t = threadIdx.x;
        const uint halft = t / 2;
        uint i = begin + (blockIdx.x * blockDim.x + threadIdx.x);
        if (i >= end) {
          return;
        }
        
        ReducableNode<D> node = read<D>(i - 1, pNode, pMinB, pDiam);
        i >>= 1;
        
        if (t % 2 == 1) {
          reductionNodes[halft] = node;
        }
        __syncthreads();
        if (t % 2 == 0) {
          node = reduce<D>(node, reductionNodes[halft]);
          if (node.node.w == 0.f) {
            reductionNodes[halft] = read<D>(i - 1, pNode, pMinB, pDiam);
          } else {
            reductionNodes[halft] = node;
            write<D>(i - 1, node, pNode, pMinB, pDiam);
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
            node = reduce<D>(reductionNodes[halft], reductionNodes[halft + (mod / 4)]);
            if (node.node.w == 0.f) {
              reductionNodes[halft] = read<D>(i - 1, pNode, pMinB, pDiam);
            } else {
              reductionNodes[halft] = node;
              write<D>(i - 1, node, pNode, pMinB, pDiam);
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
          node = reduce<D>(reductionNodes[0], reductionNodes[halfBlockDim / 2]);
          if (node.node.w > 0.f) {
            write<D>(i - 1, node, pNode, pMinB, pDiam);
          }
        }
      }

      // Explicit template instantiations
      template __global__
      void kernConstrBbox<2>(
        BVHLayout, uint, uint, uint, uint, vec<2>*, float4*, vec<2>*, float4*);
      template __global__
      void kernConstrBbox<3>(
        BVHLayout, uint, uint, uint, uint, vec<3>*, float4*, vec<3>*, float4*);

      template <>
      __global__
      void kernConstrCleanup<2>(
        uint n, float4 *pNode, vec<2> *pMinB, float4 *pDiam)
      {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) {
          return;
        }
        if (pNode[i].w == 0) {
          pMinB[i] = make_float2(0);
          pDiam[i] = make_float4(0);
        }
      }

      template <>
      __global__
      void kernConstrCleanup<3>(
        uint n, float4 *pNode, vec<3> *pMinB, float4 *pDiam)
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