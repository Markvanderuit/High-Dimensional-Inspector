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

#include <numeric>
#include <vector>
#include <cub/cub.cuh>
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_sort.h"

typedef unsigned uint;

namespace hdi {
  namespace dr {
    BVHSorter::BVHSorter() : _isInit(false) { }

    BVHSorter::~BVHSorter() {
      if (_isInit) {
        destr();
      }
    }

    void BVHSorter::init(GLuint keysUnsortedBuffer,
                            GLuint keysSortedBuffer, 
                            GLuint idxSortedBuffer, 
                            unsigned n,
                            unsigned lvls) {
      _nPos = n;
      _nLvls = lvls;

      // Fire up cub::RadixSort to figure out temporary memory size in bytes
      cub::DeviceRadixSort::SortPairs<uint, float4>(
        nullptr, _tempSize, nullptr, nullptr, nullptr, nullptr, _nPos
      );
      
      // Initialize buffers
      cudaMalloc(&_buffers(BufferType::eTemp), _tempSize);
      cudaMalloc(&_buffers(BufferType::eIdxUnsorted), _nPos * sizeof(uint));

      // Initialize interop resources
      _interops(InteropBufferType::eKeysUnsorted).init(keysUnsortedBuffer, InteropType::eReadOnly);
      _interops(InteropBufferType::eKeysSorted).init(keysSortedBuffer, InteropType::eWriteDiscard);
      _interops(InteropBufferType::eIdxSorted).init(idxSortedBuffer, InteropType::eWriteDiscard);

      // Fill unsorted indices buffer with an increasing range of sorted integers
      {
        std::vector<uint> v(_nPos);
        std::iota(v.begin(), v.end(), 0u);
        cudaMemcpy(_buffers(BufferType::eIdxUnsorted),
          v.data(), sizeof(uint) * v.size(), cudaMemcpyHostToDevice);
      }

      _isInit = true;
    }

    void BVHSorter::destr() {
      for (auto &interop : _interops) {
        interop.destr();
      }
      for (auto &buffer : _buffers) {
        cudaFree(buffer);
      }
      _isInit = false;
    }

    void BVHSorter::compute() {
      mapResources(_interops.size(), _interops.data());
      
      const int msb = 30; // exclusive
      const int lsb = msb - _nLvls;// (msb - 1) - _nLvls; // inclusive
      
      cub::DeviceRadixSort::SortPairs<uint, uint>(
        (void *) _buffers(BufferType::eTemp), 
        _tempSize, 
        (uint *) _interops(InteropBufferType::eKeysUnsorted).ptr(),
        (uint *) _interops(InteropBufferType::eKeysSorted).ptr(),
        (uint *) _buffers(BufferType::eIdxUnsorted),
        (uint *) _interops(InteropBufferType::eIdxSorted).ptr(),
        (int) _nPos, 
        lsb, msb
      );

      unmapResources(_interops.size(), _interops.data());
    }
  }
}