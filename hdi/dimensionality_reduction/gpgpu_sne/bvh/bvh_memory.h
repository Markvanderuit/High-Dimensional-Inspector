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
#include <type_traits>
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

template <typename E>
constexpr typename std::underlying_type<E>::type to_ul(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

namespace hdi {
  namespace dr {  
    namespace bvh {
      class InteropResource {
      public:
        InteropResource();
        ~InteropResource();

        void init(GLuint handle);
        void destr();

        void map();
        void unmap();
        void *ptr();

      private:
        bool _isInit;
        bool _isMapped;
        GLuint _glHandle;
        struct cudaGraphicsResource * _cuHandle;
        void * _cuPtr;
      };

      class BVHExtMemr {
      public:
        enum class MemrType {
          eNode,  // 4 x float
          eMinB,  // 4 x float
          eMaxB,  // 4 x float
          eMass,  // 1 x uint

          size    // static size, eg to_underlying(MemrType::size)
        };

        BVHExtMemr();
        ~BVHExtMemr();

        void init(const BVHLayout &layout);
        void destr();

        // OpenGL buffer access
        GLuint bufferHandle() const;
        void bindBuffer(MemrType area, GLuint index) const;

        // CUDA resource mapping
        void map();
        void unmap();

        // Area mapping
        size_t memrSize(MemrType area) const;
        size_t memrOffset(MemrType area) const;
        void *ptr(MemrType area);

      private:
        bool _isInit;
        bool _isMapped;
        GLuint _glHandle;
        struct cudaGraphicsResource *_cuHandle;
        void * _cuPtr;
        std::array<size_t, to_underlying(MemrType::size)> _memrSizes;
        std::array<size_t, to_underlying(MemrType::size)> _memrOffsets;
      };

      class BVHIntMemr {
      public:
        enum class MemrType {
          eTemp,      // ..., work space for sort
          eMortonIn,  // 1 x uint, unsorted morton codes
          eMortonOut, // 1 x uint, sorted morton codes

          size        // static size, eg to_underlying(MemrType::size)
        };

        BVHIntMemr();
        ~BVHIntMemr();

        void init(const BVHLayout &layout, size_t tempSize);
        void destr();

        size_t memrSize(MemrType type);
        size_t memrOffset(MemrType type);
        void *ptr(MemrType type);
      
      private:
        bool _isInit;
        void *_ptr;
        std::array<size_t, to_underlying(MemrType::size)> _memrSizes;
        std::array<size_t, to_underlying(MemrType::size)> _memrOffsets;
      };
    }
  }
}