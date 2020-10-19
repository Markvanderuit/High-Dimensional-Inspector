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

#include <vector>
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/interop.h"
#include <cuda_gl_interop.h> // include last to prevent double OpenGL includes from GLFW/QT

typedef unsigned uint;

 namespace hdi {
  namespace dr {
    InteropBuffer::InteropBuffer()
    : _isInit(false), _isMapped(false) { }

    InteropBuffer::~InteropBuffer() {
      if (_isInit) {
        if (_isMapped) {
          unmap();
        }
        destr();
      }
    }
    
    void InteropBuffer::init(GLuint handle, InteropType interop) {
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

    void InteropBuffer::destr() {
      ASSERT_CU(cudaGraphicsUnregisterResource(_cuHandle));
      _isInit = false;
    }

    void InteropBuffer::map() {
      ASSERT_CU(cudaGraphicsMapResources(1, &_cuHandle));
      ASSERT_CU(cudaGraphicsResourceGetMappedPointer(&_cuPtr, nullptr, _cuHandle));
      _isMapped = true;
    }
    
    void InteropBuffer::unmap() {
      ASSERT_CU(cudaGraphicsUnmapResources(1, &_cuHandle));
      _isMapped = false;
    }
    
    void * InteropBuffer::ptr() {
      return _cuPtr;
    }

    void mapResources(size_t count, InteropBuffer *resources) {
      // Collect to-be-mapped resource handles
      std::vector<cudaGraphicsResource *> handles(count);
      for (size_t i = 0; i < count; i++) {
        handles[i] = resources[i]._cuHandle;
      }

      // Map all resources
      ASSERT_CU(cudaGraphicsMapResources((int) count, handles.data()));

      // Generate mapped pointers for each resource
      for (size_t i = 0; i < count; i++) {
        resources[i]._cuHandle = handles[i];
        ASSERT_CU(cudaGraphicsResourceGetMappedPointer(
          &resources[i]._cuPtr,
          nullptr,
          resources[i]._cuHandle
        ));
        resources[i]._isMapped = true;
      }
    }

    void unmapResources(size_t count, InteropBuffer *resources) {
      // Collect to-be-unmapped resource handles
      std::vector<cudaGraphicsResource *> handles(count);
      for (size_t i = 0; i < count; i++) {
        handles[i] = resources[i]._cuHandle;
      }

      // Unmap all resources
      ASSERT_CU(cudaGraphicsUnmapResources((int) count, handles.data()));
      for (size_t i = 0; i < count; i++) {
        resources[i]._cuHandle = handles[i];
        resources[i]._cuPtr = nullptr;
        resources[i]._isMapped = false;
      }
    }
  }
}