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
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_memory.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include <cuda_gl_interop.h> // include last to prevent double OpenGL includes from GLFW/QT

template <typename genAType, typename genBType>
inline
genAType ceilDiv(genAType n, genBType div) {
  return (n + div - 1) / div;
}

template <typename genAType, typename genBType>
inline
genAType align(genAType n, genBType as) {
  return ceilDiv(n, as) * as;
}

constexpr unsigned aligned_size(unsigned d) {
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

      void InteropResource::init(GLuint handle) {
        _glHandle = handle;
        cudaGraphicsGLRegisterBuffer(&_cuHandle, _glHandle, cudaGraphicsMapFlagsNone);
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
      : _isInit(false), _isMapped(false) { }
      
      template <unsigned D>
      BVHExtMemr<D>::~BVHExtMemr() {
        if (_isInit) {
          destr();
        }
      }

      template <unsigned D>
      void BVHExtMemr<D>::init(const BVHLayout &layout) {
        // Specify memory area sizes
        _memrSizes[to_ul(MemrType::eNode)] = layout.nNodes * sizeof(float4);
        _memrSizes[to_ul(MemrType::eMinB)] = layout.nNodes * aligned_size(D);
        _memrSizes[to_ul(MemrType::eDiam)] = layout.nNodes * aligned_size(D);
        _memrSizes[to_ul(MemrType::ePos)] = layout.nPos * aligned_size(D);
        _memrSizes[to_ul(MemrType::eIdx)] = layout.nNodes * sizeof(uint);
        
        // Specify memory area offsets aligned to 16 byte endings
        _memrOffsets[to_ul(MemrType::eNode)] = 0;
        _memrOffsets[to_ul(MemrType::eMinB)] = memrOffset(MemrType::eNode) + align(memrSize(MemrType::eNode), sizeof(float4));
        _memrOffsets[to_ul(MemrType::eDiam)] = memrOffset(MemrType::eMinB) + align(memrSize(MemrType::eMinB), sizeof(float4));
        _memrOffsets[to_ul(MemrType::ePos)] = memrOffset(MemrType::eDiam) + align(memrSize(MemrType::eDiam), sizeof(float4));
        _memrOffsets[to_ul(MemrType::eIdx)] = memrOffset(MemrType::ePos) + align(memrSize(MemrType::ePos), sizeof(float4));
        
        // Create buffer object to hold entire memory area
        size_t size = memrOffset(MemrType::eIdx) + memrSize(MemrType::eIdx);//align(memrSize(MemrType::eIdx), aligned_size(D));
        glCreateBuffers(1, &_glHandle);
        glNamedBufferStorage(_glHandle, size, nullptr, GL_DYNAMIC_STORAGE_BIT);

        // Register buffer as CUDA-OpenGL interopability resources
        cudaGraphicsGLRegisterBuffer(&_cuHandle, _glHandle, cudaGraphicsMapFlagsNone);
        
        ASSERT_GL("BVHExtMemr::init()");
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
        return _memrOffsets[to_ul(type)];
      }

      template <unsigned D>
      size_t BVHExtMemr<D>::memrSize(MemrType type) const {
        return _memrSizes[to_ul(type)];
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
      : _isInit(false) { }

      BVHIntMemr::~BVHIntMemr() {
        if (_isInit) {
          destr();
        }        
      }

      void BVHIntMemr::init(const BVHLayout &layout, size_t tempSize) {
        // Specify memory area sizes
        _memrSizes[to_ul(MemrType::eTemp)] = tempSize;
        _memrSizes[to_ul(MemrType::eMortonIn)] = sizeof(uint) * layout.nPos;
        _memrSizes[to_ul(MemrType::eMortonOut)] = sizeof(uint) * layout.nPos;
        _memrSizes[to_ul(MemrType::eIdxIn)] = sizeof(uint) * layout.nPos;
        _memrSizes[to_ul(MemrType::eIdxOut)] = sizeof(uint) * layout.nPos;

        // Specify memory area offsets
        _memrOffsets[to_ul(MemrType::eTemp)] = 0;
        _memrOffsets[to_ul(MemrType::eMortonIn)] = memrOffset(MemrType::eTemp) + align(memrSize(MemrType::eTemp), 16);
        _memrOffsets[to_ul(MemrType::eMortonOut)] = memrOffset(MemrType::eMortonIn) + align(memrSize(MemrType::eMortonIn), 16);
        _memrOffsets[to_ul(MemrType::eIdxIn)] = memrOffset(MemrType::eMortonOut) + align(memrSize(MemrType::eMortonOut), 16);
        _memrOffsets[to_ul(MemrType::eIdxOut)] = memrOffset(MemrType::eIdxIn) + align(memrSize(MemrType::eIdxIn), 16);
        
        // Allocate memory to cover all memory areas
        size_t size = memrOffset(MemrType::eIdxOut) + align(memrSize(MemrType::eIdxOut), 16);
        cudaMalloc(&_ptr, size);

        _isInit = true;
      }

      void BVHIntMemr::destr() {
        cudaFree(_ptr);
        _isInit = false;
      }

      size_t BVHIntMemr::memrSize(MemrType type) {
        return _memrSizes[to_ul(type)];
      }

      size_t BVHIntMemr::memrOffset(MemrType type) {
        return _memrOffsets[to_ul(type)];
      }

      void * BVHIntMemr::ptr(MemrType type) {
        return (void *) ((char *) _ptr + memrOffset(type));
      }
    }
  }
}
