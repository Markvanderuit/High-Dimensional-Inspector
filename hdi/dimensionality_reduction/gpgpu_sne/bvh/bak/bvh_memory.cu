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

#include <algorithm>
#include <stdexcept>
#include <glad/glad.h>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_memory.h"
#include <cuda_gl_interop.h> // include last to prevent double OpenGL includes from GLFW/QT

template <typename genAType, typename genBType>
constexpr inline
genAType ceilDiv(genAType n, genBType div) {
  return (n + div - 1) / div;
}

template <typename genAType, typename genBType>
constexpr inline
genAType align(genAType n, genBType as) {
  return ceilDiv(n, as) * as;
}

constexpr inline
unsigned aligned_size(unsigned d) {
  if (d == 2) {
    return sizeof(float2);
  } else {
    return sizeof(float4);
  }
}

namespace hdi {
  namespace dr {  
    namespace bvh {
      /*
        InteropResource functions
      */

      InteropResource::InteropResource()
      : _isInit(false), _isMapped(false) { }

      InteropResource::~InteropResource() {
        if (_isInit) {
          if (_isMapped) {
            unmap();
          }
          destr();
        }
      }

      void InteropResource::init(GLuint handle, InteropType interop) {
        _glHandle = handle;
        uint flag;
        switch (interop) {
          case InteropType::eNone:
            flag = cudaGraphicsRegisterFlagsNone;  
            break;
          case InteropType::eReadOnly:
            flag = cudaGraphicsRegisterFlagsReadOnly;  
            break;
          case InteropType::eWriteDiscard:
            flag = cudaGraphicsRegisterFlagsWriteDiscard;  
            break;
        };
        ASSERT_CU(cudaGraphicsGLRegisterBuffer(&_cuHandle, _glHandle, flag));
        _isInit = true;
      }

      void InteropResource::destr() {
        cudaGraphicsUnregisterResource(_cuHandle);
        _isInit = false;
      }

      void InteropResource::map() {
        cudaGraphicsMapResources(1, &_cuHandle);
        cudaGraphicsResourceGetMappedPointer(&_cuPtr, nullptr, _cuHandle);
        _isMapped = true;
      }

      void InteropResource::unmap() {
        cudaGraphicsUnmapResources(1, &_cuHandle);
        _isMapped = false;
      }

      void * InteropResource::ptr() {
        return _cuPtr;
      }

      /*
        BVHExtMemr functions
      */

      template <unsigned D>
      BVHExtMemr<D>::BVHExtMemr()
      : _isInit(false), _isMapped(false), _logger(nullptr) { }
      
      template <unsigned D>
      BVHExtMemr<D>::~BVHExtMemr() {
        if (_isInit) {
          destr();
        }
      }

      template <unsigned D>
      void BVHExtMemr<D>::init(const BVHLayout &layout) {
        // Specify memory area sizes
        _memrSizes(MemrType::eNode) = layout.nNodes * sizeof(float4);
        _memrSizes(MemrType::eDiam) = layout.nNodes * sizeof(float4);
        _memrSizes(MemrType::eMinB) = layout.nNodes * aligned_size(D);
        _memrSizes(MemrType::ePos) = layout.nPos * aligned_size(D);
        
        // Specify memory area offsets aligned to 16 byte intervals
        _memrOffsets(MemrType::eNode) = 0;
        _memrOffsets(MemrType::eDiam) = memrOffset(MemrType::eNode) + align(memrSize(MemrType::eNode), sizeof(float4));
        _memrOffsets(MemrType::eMinB) = memrOffset(MemrType::eDiam) + align(memrSize(MemrType::eDiam), sizeof(float4));
        _memrOffsets(MemrType::ePos) = memrOffset(MemrType::eMinB) + align(memrSize(MemrType::eMinB), sizeof(float4));

        // Create buffer object to hold entire memory area
        size_t size = memrOffset(MemrType::ePos) + align(memrSize(MemrType::ePos), sizeof(float4));
        glCreateBuffers(1, &_glHandle);
        glNamedBufferStorage(_glHandle, size, nullptr, 0);
        ASSERT_GL("BVHExtMemr::init");

        // Register buffer as CUDA-OpenGL interopability resources
        ASSERT_CU(cudaGraphicsGLRegisterBuffer(&_cuHandle, _glHandle, cudaGraphicsRegisterFlagsNone));
        
        utils::secureLogValue(_logger, "   BvhExtMemr", "2 x " + std::to_string(size / 1'000'000) + "mb");

        _isInit = true;
      }

      template <unsigned D>
      void BVHExtMemr<D>::destr() {
        cudaGraphicsUnregisterResource(_cuHandle);
        glDeleteBuffers(1, &_glHandle);

        _isInit = false;
      }
      
      template <unsigned D>
      size_t BVHExtMemr<D>::memrOffset(MemrType type) const {
        return _memrOffsets(type);
      }

      template <unsigned D>
      size_t BVHExtMemr<D>::memrSize(MemrType type) const {
        return _memrSizes(type);
      }
      
      template <unsigned D>
      GLuint BVHExtMemr<D>::bufferHandle() const {
        return _glHandle;
      }

      template <unsigned D>
      void BVHExtMemr<D>::bindBuffer(MemrType type, GLuint index) const {
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, _glHandle, memrOffset(type), memrSize(type)); 
      }
      
      template <unsigned D>
      void BVHExtMemr<D>::map() {
        cudaGraphicsMapResources(1, &_cuHandle);
        cudaGraphicsResourceGetMappedPointer(&_cuPtr, nullptr, _cuHandle);
        _isMapped = true;
      }
      
      template <unsigned D>
      void BVHExtMemr<D>::unmap() {
        cudaGraphicsUnmapResources(1, &_cuHandle);
        _isMapped = false;
      }

      template <unsigned D>
      void* BVHExtMemr<D>::ptr(MemrType type) {
        return (void *) ((char *) _cuPtr + memrOffset(type));
      }
      
      // Explicit template instantiations
      template class BVHExtMemr<2>;
      template class BVHExtMemr<3>;

      /*
        BVHIntMemr functions
      */

      BVHIntMemr::BVHIntMemr()
      : _isInit(false), _logger(nullptr) { }

      BVHIntMemr::~BVHIntMemr() {
        if (_isInit) {
          destr();
        }        
      }

      void BVHIntMemr::init(const BVHLayout &layout, size_t tempSize) {
        // Specify memory area sizes
        _memrSizes(MemrType::eTemp) = tempSize;
        _memrSizes(MemrType::eMortonIn) = sizeof(uint) * layout.nPos;
        _memrSizes(MemrType::eMortonOut) = sizeof(uint) * layout.nPos;
        _memrSizes(MemrType::eIdxIn) = sizeof(uint) * layout.nPos;
        _memrSizes(MemrType::eIdxOut) = sizeof(uint) * layout.nPos;

        // Specify memory area offsets
        _memrOffsets(MemrType::eTemp) = 0;
        _memrOffsets(MemrType::eMortonIn) = memrOffset(MemrType::eTemp) + align(memrSize(MemrType::eTemp), 16);
        _memrOffsets(MemrType::eMortonOut) = memrOffset(MemrType::eMortonIn) + align(memrSize(MemrType::eMortonIn), 16);
        _memrOffsets(MemrType::eIdxIn) = memrOffset(MemrType::eMortonOut) + align(memrSize(MemrType::eMortonOut), 16);
        _memrOffsets(MemrType::eIdxOut) = memrOffset(MemrType::eIdxIn) + align(memrSize(MemrType::eIdxIn), 16);
        
        // Allocate memory to cover all memory areas
        size_t size = memrOffset(MemrType::eIdxOut) + align(memrSize(MemrType::eIdxOut), 16);
        ASSERT_CU(cudaMalloc(&_ptr, size));
        utils::secureLogValue(_logger, "   BVHIntMemr", std::to_string(size / 1'000'000) + "mb");

        _isInit = true;
      }

      void BVHIntMemr::destr() {
        cudaFree(_ptr);
        _isInit = false;
      }

      size_t BVHIntMemr::memrSize(MemrType type) {
        return _memrSizes(type);
      }

      size_t BVHIntMemr::memrOffset(MemrType type) {
        return _memrOffsets(type);
      }

      void * BVHIntMemr::ptr(MemrType type) {
        return (void *) ((char *) _ptr + memrOffset(type));
      }
    }
  }
}
