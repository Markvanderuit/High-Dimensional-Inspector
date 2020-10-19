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
#include <cub/cub.cuh>
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/interop.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/cub_interop.h"

namespace hdi {
  namespace dr {
    CUBInterop::CUBInterop()
    : _isInit(false),
      _isMapped(false),
      _tempSize(0)
    { }

    CUBInterop::~CUBInterop()
    {
      if (_isInit) {
        destr();
      }
    }

    void CUBInterop::init()
    {
      _isMapped = false;
      _isInit = true;
    }

    void CUBInterop::destr()
    {
      if (!_isInit) {
        return;
      }

      if (_isMapped) {
        unmap();
      }

      for (auto &interop : _interops) {
        interop.destr();
      }
      
      for (auto &buffer : _buffers) {
        if (buffer) cudaFree(buffer);
      }
    }

    void CUBInterop::map()
    {
      if (!_isInit) {
        return;
      }

      mapResources(_interops.size(), _interops.data());
      _isMapped = true;
    }

    void CUBInterop::unmap() {
      if (!_isInit) {
        return;
      }

      unmapResources(_interops.size(), _interops.data());
      _isMapped = false;
    }

    enum InteropPairSorterBufferTypes {
      E_PAIR_SORTER_TEMP            = 0,
      E_PAIR_SORTER_VALUES_IN       = 1,

      E_PAIR_SORTER_BUFFER_LENGTH  = 2
    };

    enum InteropPairSorterInteropTypes {
      E_PAIR_SORTER_KEYS_IN         = 0,
      E_PAIR_SORTER_KEYS_OUT        = 1,
      E_PAIR_SORTER_VALUES_OUT      = 2,
      
      E_PAIR_SORTER_INTEROP_LENGTH  = 3
    };

    void InteropPairSorter::init(GLuint keysIn, GLuint keysOut, GLuint valuesOut, uint maxn) {
      CUBInterop::init();

      // Calling DeviceRadixSort::SortPairs without parameters returns the required temp
      // memory size in tempSize
      cub::DeviceRadixSort::SortPairs<uint, uint>(
        nullptr, _tempSize, nullptr, nullptr, nullptr, nullptr, maxn
      );

      // Initialize buffers and interops
      _buffers.resize(E_PAIR_SORTER_BUFFER_LENGTH);
      _interops.resize(E_PAIR_SORTER_INTEROP_LENGTH);
      cudaMalloc(&_buffers[E_PAIR_SORTER_TEMP], _tempSize);
      cudaMalloc(&_buffers[E_PAIR_SORTER_VALUES_IN], maxn * sizeof(uint));
      _interops[E_PAIR_SORTER_KEYS_IN].init(keysIn, InteropType::eReadOnly);
      _interops[E_PAIR_SORTER_KEYS_OUT].init(keysOut, InteropType::eWriteDiscard);
      _interops[E_PAIR_SORTER_VALUES_OUT].init(valuesOut, InteropType::eWriteDiscard);

      // Fill _buffers[E_PAIR_SORTER_VALUES_IN] with iota
      {
        std::vector<uint> buffer(maxn);
        std::iota(buffer.begin(), buffer.end(), 0u);
        cudaMemcpy(_buffers[E_PAIR_SORTER_VALUES_IN],
          buffer.data(), buffer.size() * sizeof(uint), cudaMemcpyHostToDevice);
      }
    }

    void InteropPairSorter::sort(uint n, uint bits) {
      const bool doMapping = !_isMapped;
      const int msb = 30;
      const int lsb = msb - bits;
      
      if (doMapping) {
        map();
      }

      cub::DeviceRadixSort::SortPairs<uint, uint>(
        (void *) _buffers[E_PAIR_SORTER_TEMP],
        _tempSize,
        (uint *) _interops[E_PAIR_SORTER_KEYS_IN].ptr(),
        (uint *) _interops[E_PAIR_SORTER_KEYS_OUT].ptr(),
        (uint *) _buffers[E_PAIR_SORTER_VALUES_IN],
        (uint *) _interops[E_PAIR_SORTER_VALUES_OUT].ptr(),
        (int) n,
        lsb, msb
      );

      if (doMapping) {
        unmap();
      }
    }
  }
}
 