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
#include <bitset>
#include <cub/cub.cuh>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_constr.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_utils.h"

/**
 * Integer division of n by div, but rounds up instead of down.
 * 
 * Only valid for positive numbers n > 0 and div > 0. Suspect to overflow
 * for very large n and div.
 * Source: https://stackoverflow.com/questions/2745074/fast-\ceiling-of-an-integer-division-in-c-c
 */
template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

#define gpuAssert(ans) { gpuAssertDetail((ans), __FILE__, __LINE__); }
inline void gpuAssertDetail(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"gpuAssert\t%s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
        exit(code);
      }
   }
}

namespace hdi {
  namespace dr {
    namespace bvh {
      void BVHConstr::init(BVHLayout layout, size_t &ptrSize)
      {
        _layout = layout;
        _isInit = true;

        // Obtain _pTemp storage required size in byte
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          nullptr, _tempSize, nullptr, nullptr, nullptr, nullptr, _layout.nPos
        );

        // Allocate internal storage
        cudaMalloc(&_pIntr[INTR_TMP], _tempSize);
        cudaMalloc(&_pIntr[INTR_MORTON_I], _layout.nPos * sizeof(uint));
        cudaMalloc(&_pIntr[INTR_MORTON_O], _layout.nPos * sizeof(uint));
        cudaMalloc(&_pIntr[INTR_MIN_BND], (_layout.nPos + _layout.nLeaves) * sizeof(float4));
        cudaMalloc(&_pIntr[INTR_MAX_BND], (_layout.nPos + _layout.nLeaves) * sizeof(float4));

        // Write back required external storage size in bytes in ref. ptrSize
        {
          // Required sizes of four memory blocks
          std::size_t nodeSize = sizeof(float4) * (_layout.nNodes + _layout.nLeaves);
          std::size_t massSize = sizeof(uint) * (_layout.nNodes + _layout.nLeaves);
          std::size_t idxSize = sizeof(uint) * _layout.nLeaves;
          std::size_t posSize = sizeof(float4) * _layout.nPos;

          // Align uint blocks to 16 bytes to prevent potential misalignment
          massSize = ceilDiv(massSize, sizeof(float4)) * sizeof(float4);
          idxSize = ceilDiv(idxSize, sizeof(float4)) * sizeof(float4);
          
          // Required size is total of all memory blocks combined
          ptrSize = nodeSize + massSize + idxSize + posSize;
        }

        for (auto &timer : _timers) {
          timer.init();
        }
      }

      void BVHConstr::destr()
      {
        if (_isInit) {
          for (auto &timer : _timers) {
            timer.destr();
          }
          for (void *ptr : _pIntr) {
            cudaFree(ptr);
          }
          _isInit = false;
        }
      }

      void BVHConstr::compute(Bounds3D bounds, void *pEmb, void *pBvh, bool writeLog)
      {
        // Update offset ptrs given that overlying pBvh may very well change
        {
          // Offsets in bytes of last three memory blocks
          std::size_t nodeOffset = sizeof(float4) * (_layout.nNodes + _layout.nLeaves);
          std::size_t massOffset = sizeof(uint) * (_layout.nNodes + _layout.nLeaves);
          std::size_t idxOffset = sizeof(uint) * _layout.nLeaves;

          // Align uint blocks to 16 bytes to prevent potential misalignment
          massOffset = ceilDiv(massOffset, sizeof(float4)) * sizeof(float4);
          idxOffset = ceilDiv(idxOffset, sizeof(float4)) * sizeof(float4);
          
          _pExtr[EXTR_NODE] = pBvh;
          _pExtr[EXTR_MASS] = static_cast<char *>(_pExtr[EXTR_NODE]) + nodeOffset;
          _pExtr[EXTR_IDX] = static_cast<char *>(_pExtr[EXTR_MASS]) + massOffset;
          _pExtr[EXTR_POS] = static_cast<char *>(_pExtr[EXTR_IDX]) + idxOffset;
        }
        
        // BVH bounds in float4 format
        float4 minb = make_float4(bounds.min.x, bounds.min.y, bounds.min.z, 0.f);
        float4 invr = make_float4(1.f) 
          / make_float4(bounds.range().x, bounds.range().y, bounds.range().z, 1.f);
        
        gpuAssert(cudaDeviceSynchronize());

        // Generate morton codes over unsorted embedding positions
        _timers[TIMR_MORTON].tick();
        kernConstrMorton<<<ceilDiv(_layout.nPos, 256u), 256u>>>(
          _layout, minb, invr,
          (float4 *)  pEmb,
          (uint *)    _pIntr[INTR_MORTON_I]
        );
        _timers[TIMR_MORTON].tock();

        gpuAssert(cudaDeviceSynchronize());

        // Radix sort embedding positions along generated morton codes
        _timers[TIMR_SORT].tick();
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          (void *)    _pIntr[INTR_TMP], 
          _tempSize, 
          (uint *)    _pIntr[INTR_MORTON_I], 
          (uint *)    _pIntr[INTR_MORTON_O], 
          (float4 *)  pEmb, 
          (float4 *)  _pExtr[EXTR_POS], 
          _layout.nPos, 
          0, 29
        );
        _timers[TIMR_SORT].tock();
        
        gpuAssert(cudaDeviceSynchronize());

        // Generate leaf nodes over sorted embedding positions
        _timers[TIMR_LEAVES].tick();
        kernConstrLeaves<<<ceilDiv(_layout.nLeaves, 256u), 256u>>>(
          _layout, 
          (float4 *)  _pExtr[EXTR_POS], 
          (float4 *)  _pExtr[EXTR_NODE],
          (uint *)    _pExtr[EXTR_MASS], 
          (uint *)    _pExtr[EXTR_IDX], 
          (float4 *)  _pIntr[INTR_MIN_BND], 
          (float4 *)  _pIntr[INTR_MAX_BND]
        );
        _timers[TIMR_LEAVES].tock();
        
        gpuAssert(cudaDeviceSynchronize());

        // Generate non-leaf nodes through reduction of leaf nodes in
        // probably a few passes
        _timers[TIMR_NODES].tick();
        {
          const uint nThreads = 256;
          const uint logk = static_cast<uint>(std::log2(_layout.kNode));
          const int iter = static_cast<int>(std::log2(nThreads) / logk);
          for (int lbegin = _layout.nLvls - 1; lbegin >= 0; lbegin -= iter) {
            const int lend = std::max(0, lbegin - (iter + 1));
            
            kernConstrNodes<<<256, 256>>>(
              _layout, 
              lbegin, lend, 
              (float4 *)  _pExtr[EXTR_NODE],
              (uint *)    _pExtr[EXTR_MASS],
              (float4 *)  _pIntr[INTR_MIN_BND], 
              (float4 *)  _pIntr[INTR_MAX_BND]
            );
          }
        }
        _timers[TIMR_NODES].tock();
        
        gpuAssert(cudaDeviceSynchronize());
        
        for (auto &timer : _timers) {
          timer.poll();
        }

        if (writeLog) {
          utils::secureLog(_logger, "BVH construction");
          _timers[TIMR_MORTON].log(_logger, "  Morton");
          _timers[TIMR_SORT].log(_logger, "  Sorting");
          _timers[TIMR_LEAVES].log(_logger, "  Leaves");
          _timers[TIMR_NODES].log(_logger, "  Nodes");
          utils::secureLog(_logger, "");
        }
      }

      BVHConstr::BVHConstr()
      : _isInit(false)
      {
        // ...
      }

      BVHConstr::~BVHConstr()
      {
        if (_isInit) {
          destr();
        }
      }
    }
  }
}