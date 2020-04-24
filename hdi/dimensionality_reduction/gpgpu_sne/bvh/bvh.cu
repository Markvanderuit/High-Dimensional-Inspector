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

#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"
#include <cuda_gl_interop.h> // include last to prevent double OpenGL includes from GLFW/QT

 namespace hdi {
  namespace dr {  
    namespace bvh {
      BVH::BVH()
      {
        // ...
      }

      BVH::~BVH()
      {
        if (_isInit) {
          destr();
        }
      }
      
      void BVH::init(const TsneParameters &params, GLuint posBuffer, unsigned nPos)
      {
        _params = params;
        _iter = 0;
        _layout = BVHLayout(32, 8, nPos); // TODO Extract fan-out parameters
        _isInit = true;

        std::cout << to_string(_layout) << std::endl;

        // Query BVH constructor for required buffer size in bytes
        size_t buffSize = 0u;
        _constr.init(_layout, buffSize);

        // Create buffer object to hold entire BVH structure in a single block of memory
        glCreateBuffers((GLsizei) _buffers.size(), _buffers.data());
        glNamedBufferStorage(_buffers[BUFF_BVH], buffSize, nullptr, GL_DYNAMIC_STORAGE_BIT);

        // Register buffers as CUDA-OpenGL interopability resources
        cudaGraphicsGLRegisterBuffer(&_resources[RSRC_EMB], 
          posBuffer, cudaGraphicsMapFlagsReadOnly);
        cudaGraphicsGLRegisterBuffer(&_resources[RSRC_BVH], 
          _buffers[BUFF_BVH], cudaGraphicsMapFlagsNone);
      }

      void BVH::destr()
      {
        if (_isInit) {
          _constr.destr();
          for (auto& rsrc : _resources) {
            cudaGraphicsUnregisterResource(rsrc);
          }
          glDeleteBuffers((GLsizei) _buffers.size(), _buffers.data());
          _isInit = false;
        }
      }

      void BVH::compute(const Bounds3D &bounds)
      {
        // Map graphics resources for access in CUDA
        cudaGraphicsMapResources((uint) _resources.size(), _resources.data());
        
        // Obtain mapped ptrs
        void *pEmb;
        void *pBvh;
        size_t embSize;
        size_t bvhSize;
        cudaGraphicsResourceGetMappedPointer(&pEmb, &embSize, _resources[RSRC_EMB]);
        cudaGraphicsResourceGetMappedPointer(&pBvh, &bvhSize, _resources[RSRC_BVH]);
        
        // Offload BVH construction to BVHConstr class
        _constr.compute(bounds, pEmb, pBvh, _iter == _params._iterations - 1);

        // Unmap graphics resources
        cudaGraphicsUnmapResources((uint) _resources.size(), _resources.data());

        _iter++;
      }
    }
  }
}