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

#include <vector>
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/interop.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/timer.h"

namespace hdi {
  namespace dr {
    class CUBInterop {
      using uint = unsigned;

    public:
      CUBInterop();
      ~CUBInterop();

      void destr();
      void map();
      void unmap();

    protected:
      void init();

      bool _isInit;
      bool _isMapped;
      size_t _tempSize; // All CUB primitives require this parameter
      std::vector<void *> _buffers;
      std::vector<InteropBuffer> _interops;
    };

    class InteropPairSorter : public CUBInterop {
      using uint = unsigned;

    public:
      // Specify keysOut as sorted version of keysIn
      // additionally return a "sort order" in valuesOut
      // Max nr of sortable elements must be specified beforehand
      void init(GLuint keysIn,
                GLuint keysOut,
                GLuint valuesOut,
                uint maxn);

      void sort(uint n, uint bits);
    };

    class InteropInclusiveScanner : public CUBInterop {
    using uint = unsigned;

    public:
      // Specify valuesOut as scanned result over valuesIn
      // Max nr of scannable elements must be specified beforehand
      void init(GLuint valuesIn, 
                GLuint valuesOut,
                uint maxn);
                       
      void inclusiveScan(uint n);
    };
  }
}