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

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_timer.h"

namespace hdi {
  namespace dr {  
    namespace bvh {
      class BVH3DCompute {
      public:
        BVH3DCompute();
        ~BVH3DCompute();
        
        void initialize(const TsneParameters& params, GLuint pos_buff, unsigned n);

        void clean();

        void compute(unsigned w, unsigned h, unsigned d, unsigned n,
                     GLuint pos_buff, GLuint bounds_buff, Bounds3D bounds);

      private:
        enum CUDAMemoryType {
          // Type values
          MEMR_CODES_UNSORTED,      // Device memory storing unsorted Morton codes
          MEMR_CODES_SORTED,        // Device memory storing sorted Morton codes
          MEMR_POS_SORTED,          // Device memory storing sorted embedding positions
          MEMR_SORT_TEMP,           // Device memory used for temp storage by cub::DeviceRadixSort()

          // Static nr. of type values
          CUDAMemoryTypeLength
        };

        enum GLBufferType {
          // Type values
          BUFF_BVH,                 // GL buffer object storing full BVH data except bounds

          // Static nr. of type values
          GLBufferTypeLength
        };

        enum ResourceType {
          // Type values
          RSRC_POS_UNSORTED,        // Resource providing CUDA-GL interopability for embedding pos.
          RSRC_BVH,                 // Resource providing CUDA-GL interopability for BVH data

          // Static nr. of type values
          ResourceTypeLength
        };

        enum TimerType {
          // Type values
          TIMR_CODES,               // Timer measuring morton code generation time
          TIMR_SORT,                // Timer measuring cub::DeviceRadoxSort() execution time
          TIMR_BVH_LEAVES,          // Timer measuring BVH leaf generation time
          TIMR_BVH_NODES,           // Timer measuring BVH node generation time

          // Static nr. of type values
          TimerTypeLength
        };

        std::array<GLuint, GLBufferTypeLength> _GLbuffers;
        std::array<void *, CUDAMemoryTypeLength> _CUDAMemory;
        std::array<struct cudaGraphicsResource *, ResourceTypeLength> _resources;
        std::array<CuTimer, TimerTypeLength> _timers;

        bool _isInitialized;
        int _iteration;
        size_t _memSortTempSize;
        TsneParameters _params;
        utils::AbstractLog* _logger;

      public:
        void setLogger(utils::AbstractLog* logger) {
          _logger = logger; 
        }
        
        GLuint buffer() const {
          return _GLbuffers[BUFF_BVH];
        }
      };
    }
  }
}