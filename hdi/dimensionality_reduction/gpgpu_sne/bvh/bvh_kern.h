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
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"

namespace hdi {
  namespace dr {
    namespace bvh {
      /**
       * CUDA kernel to transform 3D embedding positions into 30 bit morton codes.
       * The positions will be normalized before this transformation happens.
       * 
       * - layout   BVH layout data structure
       * - minb     float4 minimum embedding bounds
       * - invr     float4 inverse of embedding range, ie 1 / (maxb - minb)
       * - pEmb     ptr to unsorted embedding positional data, padded to float4 values.
       * - pMorton  ptr to memory in which morton codes will be stored
       */
      __global__
      void kernConstrMorton(
        BVHLayout layout, 
        float4 minb, float4 invr, 
        float4 *pEmb, uint *pMorton);

      __global__
      void kernConstrPos(
        BVHLayout layout,
        uint *idx, float4 *posIn, float4 *posOut);

      /**
       * CUDA kernel to perform subdivision of BVH nodes using sorted morton codes.
       */
      __global__
      void kernConstrSubdiv(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        uint *pMorton, uint *pIdx, float4 *pPos, float4 *pNode, float4 *pMinB, float4 *pDiam);

      __global__
      void kernConstrDataReduce(
        BVHLayout layout,
        uint level, uint levels, uint begin, uint end,
        uint *pIdx, float4 *pPos, float4 *pNode, float4 *pMinB, float4 *pDiam);

      __global__
      void kernConstrCleanup(
        uint n, float4 *pNode, float4 *pMinB, float4 *pDiam);
    }
  }
}