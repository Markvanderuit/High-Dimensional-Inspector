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
 #include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"

 namespace hdi {
  namespace dr {
    namespace bvh {
      __device__
      inline
      float4 max4f(const float4 &a, const float4 &b)
      {
        return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
      }

      __device__
      inline
      float4 min4f(const float4 &a, const float4 &b)
      {
        return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
      }

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
      void kernConstrLeaves(BVHLayout layout, 
        float4 *pPos, float4 *pNode, uint *pMass, uint *pIdx, float4 *pMinB, float4 *pMaxB)
      {
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= layout.nLeaves) {
          return;
        }

        uint idx = gid * layout.kLeaf; 
        uint mass = 0u; 

        // Check border condition to determine mass
        // as empty leaves likely exist in the current BVH structure
        int incr = static_cast<int>(layout.nPos) - static_cast<int>(idx);
        if (incr > 0) {
          mass = min(layout.kLeaf, static_cast<uint>(incr));
        }
        
        float4 center = make_float4(0);
        float4 minb = make_float4(0);
        float4 maxb = make_float4(0);
        
        // Iterate over kLeaf points to determine center of mass, min and max bounds
        if (mass > 0) {
          center = pPos[idx];
          minb = center;
          maxb = center;
        }
        for (int i = ++idx; i < idx + mass; ++i) {
          float4 pos = pPos[i];
          center += pos;
          minb = min4f(minb, pos);
          maxb = max4f(maxb, pos);
        }
        center /= static_cast<float>(max(mass, 1u));
        center.w = dot(minb, maxb);

        // Store leaf node data
        uint addr = layout.nNodes + gid;
        pNode[addr] = center;
        pMass[addr] = mass;
        pMinB[addr] = minb;
        pMaxB[addr] = maxb;
        pIdx[gid] = idx;
      }

      __device__
      void reduce(
        float4 &lnode, uint &lmass, float4 &lminb, float4 &lmaxb,
        float4 rnode, uint rmass, float4 rminb, float4 rmaxb)
      {
        const uint _mass = lmass + rmass;
        lnode = (static_cast<float>(lmass) * lnode + static_cast<float>(rmass) * rnode)
              / static_cast<float>(_mass);
        lmass = _mass;
        lminb = min4f(lminb, rminb);
        lmaxb = max4f(lmaxb, rmaxb);
        lnode.w = dot(lminb, lmaxb);
      }

      __device__
      void reduceOuter(BVHLayout layout, 
        float4 node, uint mass, float4 minB, float4 maxB,
        float4 *pNode, uint *pMass, float4 *pMinB, float4 *pMaxB)
      {
        const uint lid = threadIdx.x;
        const uint idx = lid / 2;
        const uint mod = lid % 2;
        if (mod == 0u) {
          // FIXME only works on first iteration
          pNode[idx] = node;
          pMinB[idx] = minB;
          pMaxB[idx] = maxB;
          pMass[idx] = mass;
        }
        __syncthreads();
        if (mod == 1u) {
          // ...
        }
      }

      __global__
      void kernConstrNodes(BVHLayout layout, 
        uint lbegin, uint lend,
        float4 *pNode, uint *pMass, float4 *pMinB, float4 *pMaxB)
      {
        // Declare shared memory and ptrs for different data parts
        // For kNode = 2, at most requires 26kb per thread block
        // which of course becomes smaller as kNode increases
        extern __shared__ float4 _pShared[];
        const uint nSharedValues = blockDim.x / layout.kNode; 
        float4 *pSharedNode = _pShared;
        float4 *pSharedMinB = (float4 *) &pSharedNode[nSharedValues];
        float4 *pSharedMaxB = (float4 *) &pSharedMinB[nSharedValues];
        uint *pSharedMass = (uint *) &pSharedMaxB[nSharedValues];
        
        const uint lid = threadIdx.x;
        const uint shid = threadIdx.x / layout.kNode;
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint logk = static_cast<uint>(log2(_layout.kNode));
        const uint idx = (1u << (logk * (lbegin + 1))) +  gid;
        
        float4 prevNode = pNode[idx];
        float4 prevMinB = pMinB[idx];
        float4 prevMaxB = pMaxB[idx];
        uint prevMass = pMass[idx];

        /* // Load node from previous level

        // Perform first reduction by hand
        if (lid % _layout.kNode == 0u) {
          pSharedNode[lid] = prevNode;
          pSharedMinB[lid] = prevMinB;
          pSharedMaxB[lid] = prevMaxB;
          pSharedMass[lid] = prevMass;
        }
        __syncthreads();
        for (uint i = 1; i < _layout.kNode; i++) {
          if (lid % _layout.kNode == i) {
            
          }
        } */
        
      }
    }
  }
}