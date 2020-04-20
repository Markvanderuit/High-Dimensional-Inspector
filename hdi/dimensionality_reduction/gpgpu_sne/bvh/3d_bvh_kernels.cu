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

#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/3d_bvh_kernels.h"

namespace hdi {
  namespace dr {
    namespace bvh {
      __device__
      uint expandBits(uint i)
      {
        i = (i * 0x00010001u) & 0xFF0000FFu;
        i = (i * 0x00000101u) & 0x0F00F00Fu;
        i = (i * 0x00000011u) & 0xC30C30C3u;
        i = (i * 0x00000005u) & 0x49249249u;
        return i;
      }
      
      __device__
      uint computeMortonCode(float4 f)
      {
        f = clamp(f * 1024.f, 0.f, 1023.f);
        uint x = expandBits(static_cast<uint>(f.x));
        uint y = expandBits(static_cast<uint>(f.y));
        uint z = expandBits(static_cast<uint>(f.z));
        return x * 4u + y * 2u + z;
      }

      __global__
      void kernComputeMortonCode(float4 *pPos, uint *pCode, float4 minb, float4 invr, uint n)
      {
        // Invocation id
        // Only proceed with invocs under n
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= n) {
          return;
        }
      
        // Cmpute morton code over normalized position
        const float4 pos = (pPos[gid] - minb) * invr;
        const uint code = computeMortonCode(pos);

        pCode[gid] = code;
      }

      __global__
      void kernComputeBVHLeaf(float4 *pPos, BVHNode *pNode, uint lvl, uint nPos, uint nLeaf)
      {
        // Invocation id
        // Only proceed with invocs under nr. of leaves
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= nLeaf) {
          return;
        }
        
        // Construct node, only set data if invoc under nr. of positions
        BVHNode node = BVHNode();
        if (gid < nPos) {
          const float4 pos = pPos[gid];
          node = BVHNode(
            make_float4(pos.x, pos.y, pos.z, 1.f),
            make_float4(pos.x, pos.y, pos.z, 0.f),
            make_float4(pos.x, pos.y, pos.z, 0.f)
          );
        }
        
        uint idx = (1u << lvl) - 1u + gid;
        pNode[idx] = node;
      }

      __device__
      inline
      float4 maxf(const float4 &a, const float4 &b)
      {
        return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
      }

      __device__
      inline
      float4 minf(const float4 &a, const float4 &b)
      {
        return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
      }

      __device__
      BVHNode reduce(const BVHNode &lnode, const BVHNode &rnode)
      {
        // Empty nodes are always aligned to the right
        // and we should skip reducing these
        if (rnode.v.w == 0.f) {
          return lnode;
        }

        const float3 lcom = make_float3(lnode.v.x, lnode.v.y, lnode.v.z);
        const float3 rcom = make_float3(rnode.v.x, rnode.v.y, rnode.v.z);
        
        // Mass is combined mass of children
        const float n = lnode.v.w + rnode.v.w;

        // Center of mass is average weighted by child masses
        const float3 com = (lnode.v.w * lcom + rnode.v.w * rcom) / max(n, 1.f); // clamp div-by-0
        
        // Bounds are expanded min/max bounds
        const float4 maxb =  maxf(lnode.maxb, rnode.maxb);
        float4 minb = minf(lnode.minb, rnode.minb);

        float3 t = make_float3(maxb.x, maxb.y, maxb.z) - make_float3(minb.x, minb.y, minb.z);
        minb.w = dot(t, t);
        return BVHNode {
          make_float4(com.x, com.y, com.z, n),
          minb,
          maxb
        };
      }

      __global__
      void kernComputeBVHNode(BVHNode *pNode, uint k, int lmax, int lmin)
      {
        // Declare shared memory in which to perform reduction of subtree
        extern __shared__ BVHNode shNode[];

        // Invocation id
        const uint lid = threadIdx.x;
        const uint shid = threadIdx.x / 2;
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Compute nr. of nodes in previous level (lmax + 1) depending on arity (k)
        const uint logk = uint(std::log2f(k)); // 1 for k=2, 2 for k=4, 3 for k=8, 4 for k=16...
        const uint nPrevNodes = (1u << (logk * uint(lmax + 1)));

        // Only proceed with invocs under n
        if (gid >= nPrevNodes) {
          return;
        }

        // Load node from previous level
        BVHNode prevNode = pNode[(1u << (logk * uint(lmax + 1))) - 1u + gid];

        // Perform first step of parallel reduction across k threads
        for (int i = 2; i <= k; i++) {
          if (lid % i != 0) {
            shNode[shid] = prevNode;
          }
          __syncthreads();
          if (lid % 2 == 0) {
            shNode[shid] = reduce(prevNode, shNode[shid]);
            __syncthreads();
          }
        }


        __syncthreads();
        if (lid % 2 == 0) {
          shNode[shid] = reduce(prevNode, shNode[shid]);

          // Write to global memory
          uint idx = (1u << uint(lmax)) - 1u + (gid / 2u);
          pNode[idx] = shNode[shid];
        }
        __syncthreads();

        // ...
      }

      __global__
      void kernComputeBVHNode(BVHNode *pNode, int lmax, int lmin)
      {
        // Declare shared memory in which to perform reduction of subtree
        extern __shared__ BVHNode shNode[];

        // Invocation id
        const uint lid = threadIdx.x;
        const uint shid = threadIdx.x / 2;
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint nPrevNodes = (1u << (uint(lmax) + 1u)); // nr of nodes in previous level

        // Only proceed with invocs under n
        if (gid >= nPrevNodes) {
          return;
        }
        
        // Load node from previous level
        BVHNode prevNode = pNode[(1u << (uint(lmax) + 1u)) - 1u + gid];
        
        // Perform first step of parallel reduction
        if (lid % 2 != 0) {
          shNode[shid] = prevNode;
        }
        __syncthreads();
        if (lid % 2 == 0) {
          shNode[shid] = reduce(prevNode, shNode[shid]);

          // Write to global memory
          uint idx = (1u << uint(lmax)) - 1u + (gid / 2u);
          pNode[idx] = shNode[shid];
        }
        __syncthreads();
        
        // Perform remaining steps of parallel reduction
        uint modOffset = 1;
        uint modFactor = 4;
        for (int l = lmax - 1; l >= lmin; l--) {
          if (lid % modFactor == 0) {
            shNode[shid] = reduce(shNode[shid], shNode[shid + modOffset]);

            // Write to global memory
            uint idx = (1u << uint(l)) - 1u + (gid / modFactor);
            pNode[idx] = shNode[shid];
          }
          __syncthreads();
          
          // Increase factor, offset for next level
          modOffset *= 2;
          modFactor *= 2;
        }
      }
    }
  }
}