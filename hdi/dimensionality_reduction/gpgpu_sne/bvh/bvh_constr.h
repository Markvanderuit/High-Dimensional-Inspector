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
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"

namespace hdi {
  namespace dr {
    namespace bvh {
      /**
       * CUDA-based constructor of implicit Bounding Volume Hierarchy (BVH) 
       * tailored to TSNE embeddings for fast GPGPU approximations.
       * The implicit BVH is constructed in a contiguous block of 
       * user-provided memory, and can therefore be easily moved
       * between CUDA and OpenGL for flexibility.
       * 
       * @author Mark van de Ruit
       */
      class BVHConstr {
      public:
        /**
         * Initialize the BVH constructor. Writes to ptrSize the required
         * size in bytes of memory that must be provided to the compute() function
         * when constructing the actual BVH, and allocates internal memory.
         */
        void init(BVHLayout layout,   // Object describing the BVH's layout
                        size_t &ptrSize);   // Required size of memory passed to compute()

        /**
         * Clean the BVH constructor. Deallocates internal memory.
         */
        void destr();

        /**
         * Construct the BVH over nPos embedding positions in pEmb, 
         * storing it in the memory provided in pBvh.
         */
        void compute(Bounds3D bounds, void *pEmb, void *pBvh, bool writeLog = false);

        // Base constr, destr.
        BVHConstr();
        ~BVHConstr();

      private:
        // Pointer handles for internal memory: components used during BVH construction
        enum IntrMemType {
          INTR_TMP,         // (... *)    Sort in, temporary storage, size stored in _tempSize
          INTR_MORTON_I,    // (uint *)   Sort in, unsorted morton codes
          INTR_MORTON_O,    // (uint *)   Sort out, sorted morton codes
          INTR_MIN_BND,     // (float4 *) Node minimum bounds vectors
          INTR_MAX_BND,     // (float4 *) Node maximum bounds vectors
          IntrMemTypeLength
        };

        // Pointer handles for external memory: components used for Barnes-Hut TSNE
        enum ExtrMemType {
          EXTR_NODE,        // (float4 *) Node structure: xyz = center of mass, w = diams^2
          EXTR_MASS,        // (uint *)   Node mass: nr of contained embedding positions
          EXTR_IDX,         // (uint *)   Starting idx of leaf node's contained embedding position
          EXTR_POS,         // (float4 *) Sorted embedding positions
          ExtrMemTypeLength
        };
        
        // Timer handles for kernel runtime performance measurements
        enum TimerType {
          TIMR_MORTON,      // Morton code generation timer 
          TIMR_SORT,        // Radix sort timer
          TIMR_LEAVES,      // Leaf node computation timer
          TIMR_NODES,       // Non-leaf node computation timer
          TimerTypeLength
        };

        bool _isInit;       // Is class initialized, ie is memory allocated
        BVHLayout _layout;  // Layout and size description of the BVH
        size_t _tempSize;   // Size in bytes of temp storage for sort
        utils::AbstractLog* _logger;
        std::array<CuTimer, TimerTypeLength> _timers;  // Timers for kernel perf.
        std::array<void *, IntrMemTypeLength> _pIntr;  // Ptrs to internal memory
        std::array<void *, ExtrMemTypeLength> _pExtr;  // Ptrs to external memory
        
      public:
        void setLogger(utils::AbstractLog* logger) {
          _logger = logger; 
        }
      };
    }
  }
}