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
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_memory.h"
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
        if (!_isMapped) {
          throw std::runtime_error("InteropResource::ptr() called while memory is not mapped for access");
        }
        return _cuPtr;
      }

      /*
        BVHExtMemr functions
      */

      BVHExtMemr::BVHExtMemr()
      : _isInit(false), _isMapped(false) { }
      
      BVHExtMemr::~BVHExtMemr() {
        if (_isInit) {
          destr();
        }
      }

      void BVHExtMemr::init(const BVHLayout &layout) {
        size_t float4Size = sizeof(float4);
        size_t uintSize = sizeof(uint);
        
        // Specify memory area sizes
        _memrSizes[to_underlying(MemrType::eNode)] = (layout.nNodes + layout.nLeaves) * float4Size;
        _memrSizes[to_underlying(MemrType::eMinB)] = (layout.nNodes + layout.nLeaves) * float4Size;
        _memrSizes[to_underlying(MemrType::eMaxB)] = (layout.nNodes + layout.nLeaves) * float4Size;
        _memrSizes[to_underlying(MemrType::ePos)] = (layout.nPos) * float4Size;
        _memrSizes[to_underlying(MemrType::eMass)] = (layout.nNodes + layout.nLeaves) * uintSize;
        _memrSizes[to_underlying(MemrType::eIdx)] = (layout.nLeaves) * uintSize;
        
        // Specify memory area offsets
        _memrOffsets[to_underlying(MemrType::eNode)] = 0;
        _memrOffsets[to_underlying(MemrType::eMinB)] = memrOffset(MemrType::eNode) + align(memrSize(MemrType::eNode), float4Size);
        _memrOffsets[to_underlying(MemrType::eMaxB)] = memrOffset(MemrType::eMinB) + align(memrSize(MemrType::eMinB), float4Size);
        _memrOffsets[to_underlying(MemrType::ePos)] = memrOffset(MemrType::eMaxB) + align(memrSize(MemrType::eMaxB), float4Size);
        _memrOffsets[to_underlying(MemrType::eMass)] = memrOffset(MemrType::ePos) + align(memrSize(MemrType::ePos), float4Size);
        _memrOffsets[to_underlying(MemrType::eIdx)] = memrOffset(MemrType::eMass) + align(memrSize(MemrType::eMass), float4Size);
        
        // Create buffer object to hold entire memory area
        size_t size = memrOffset(MemrType::eIdx) + align(memrSize(MemrType::eIdx), float4Size);
        glCreateBuffers(1, &_glHandle);
        glNamedBufferStorage(_glHandle, size, nullptr, GL_DYNAMIC_STORAGE_BIT);

        // Register buffer as CUDA-OpenGL interopability resources
        cudaGraphicsGLRegisterBuffer(&_cuHandle, _glHandle, cudaGraphicsMapFlagsNone);

        _isInit = true;
      }

      void BVHExtMemr::destr() {
        cudaGraphicsUnregisterResource(_cuHandle);
        glDeleteBuffers(1, &_glHandle);

        _isInit = false;
      }
      
      size_t BVHExtMemr::memrOffset(MemrType type) const {
        return _memrOffsets[to_underlying(type)];
      }

      size_t BVHExtMemr::memrSize(MemrType type) const {
        return _memrSizes[to_underlying(type)];
      }
      
      GLuint BVHExtMemr::bufferHandle() const {
        return _glHandle;
      }

      void BVHExtMemr::bindBuffer(MemrType type, GLuint index) const {
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, index, _glHandle, memrOffset(type), memrSize(type)); 
      }
      
      void BVHExtMemr::map() {
        cudaGraphicsMapResources(1, &_cuHandle);
        cudaGraphicsResourceGetMappedPointer(&_cuPtr, nullptr, _cuHandle);
        _isMapped = true;
      }
      
      void BVHExtMemr::unmap() {
        cudaGraphicsUnmapResources(1, &_cuHandle);
        _isMapped = false;
      }

      void* BVHExtMemr::ptr(MemrType type) {
        if (!_isMapped) {
          throw std::runtime_error("BVHExtMemr::memPtr() called while memory is not mapped for access");
        }
        return (void *) ((char *) _cuPtr + memrOffset(type));
      }

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

        // Specify memory area offsets
        _memrOffsets[to_ul(MemrType::eTemp)] = 0;
        _memrOffsets[to_ul(MemrType::eMortonIn)] = memrOffset(MemrType::eTemp) + align(memrSize(MemrType::eTemp), 16);
        _memrOffsets[to_ul(MemrType::eMortonOut)] = memrOffset(MemrType::eMortonIn) + align(memrSize(MemrType::eMortonIn), 16);
        
        // Allocate memory to cover all memory areas
        size_t size = memrOffset(MemrType::eMortonOut) + align(memrSize(MemrType::eMortonOut), 16);
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
