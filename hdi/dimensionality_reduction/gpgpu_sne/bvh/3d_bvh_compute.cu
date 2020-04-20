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
#include <bitset>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <cub/cub.cuh>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/3d_bvh_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/3d_bvh_kernels.h"
#include <cuda_gl_interop.h> // include last to prevent double OpenGL includes from GLFW/QT
 
// constexpr unsigned k = 2; // Tree arity

namespace hdi {
  namespace dr {  
    namespace bvh {
      BVH3DCompute::BVH3DCompute()
      : _isInitialized(false),
        _logger(nullptr),
        _iteration(0)
      { }

      BVH3DCompute::~BVH3DCompute()
      {
        if (_isInitialized) {
          clean();
        }
      }

      void BVH3DCompute::initialize(const TsneParameters& params, GLuint pos_buff, unsigned n)
      {
        _params = params;
        _iteration = 0;
        
        // Compute tree size in terms of leaves and nodes - leaves, given varying input arity
        {
          // Input
          const uint _n = 10000;  // nr of points in embedding
          const uint _k = 8;      // arity

          // 1 for k=2, 2 for k=4, 3 for k=8, 4 for k=16...
          const uint logk = uint(std::log2(_k)); 
          
          // Nr of levels is logk(n)
          const uint nLvls = static_cast<uint>(std::ceil(std::log2(_n) / logk));

          const uint nLeaves = (1u << (logk * nLvls));
          uint nNodes = 0u;
          for (uint i = 0u; i < nLvls; i++) {
            nNodes |= (1u << (logk * i)); 
          }

          std::cout << nLvls << ", " << nLeaves << ", " << nNodes << std::endl;
        }

        // Compute tree size in terms of leaves and nodes minus leaves
        const uint k = 2; // Tree arity
        const uint logk = static_cast<uint>(std::log2(k)); 
        
        const uint nLvl = static_cast<uint>(std::ceil(std::log2(n) / logk));
        const uint nLeaf = (1u << (logk * nLvl));
        uint nNode = nLeaf;
        for (uint i = 0u; i < nLvl; i++) {
          nNode |= (1u << (logk * i)); 
        }

        // Allocate CUDA device memory (temp not included)
        cudaMalloc(&_CUDAMemory[MEMR_CODES_UNSORTED], n * sizeof(uint));
        cudaMalloc(&_CUDAMemory[MEMR_CODES_SORTED], n * sizeof(uint));
        cudaMalloc(&_CUDAMemory[MEMR_POS_SORTED], n * sizeof(float4));

        // Create OpenGL buffer object(s)
        glCreateBuffers((GLsizei) _GLbuffers.size(), _GLbuffers.data());
        glNamedBufferStorage(_GLbuffers[BUFF_BVH], 
                             nNode * sizeof(BVHNode), 
                             nullptr,
                             GL_DYNAMIC_STORAGE_BIT);

        // Register resources for CUDA-GL interopability
        cudaGraphicsGLRegisterBuffer(&_resources[RSRC_POS_UNSORTED],
                                    pos_buff,
                                    cudaGraphicsMapFlagsReadOnly);
        cudaGraphicsGLRegisterBuffer(&_resources[RSRC_BVH],
                                    _GLbuffers[BUFF_BVH],
                                    cudaGraphicsMapFlagsNone);
        
        // Query required size of cub::DeviceRadixSort() temp storage
        // Initialize temp storage for cub::DeviceRadixSort()
        // Requires us to obtain the required size first
        {
          // Map pos resource and obtain mapped ptr
          void *pPos;
          size_t posSize = n * sizeof(float4);
          cudaGraphicsMapResources(1, &_resources[RSRC_POS_UNSORTED]);
          cudaGraphicsResourceGetMappedPointer(&pPos, &posSize, _resources[RSRC_POS_UNSORTED]);
                                               
          // Perform DeviceRadixSort::SortPairs with no temporary storage provided
          // This writes required temporary storage size in bytes to memSortingTempSize
          cub::DeviceRadixSort::SortPairs<uint, float4>(
            nullptr,
            _memSortTempSize,
            nullptr, // (uint *) _CUDAMemory[MEMR_CODES_UNSORTED], 
            nullptr, // (uint *) _CUDAMemory[MEMR_CODES_SORTED],
            nullptr, // (float4 *) pPos,
            nullptr, // (float4 *) _CUDAMemory[MEMR_POS_SORTED], 
            n
          );

          // Unmap pos resource and allocate temporary storage
          cudaGraphicsUnmapResources(1, &_resources[RSRC_POS_UNSORTED]);
          cudaMalloc(&_CUDAMemory[MEMR_SORT_TEMP], _memSortTempSize);
        }

        // Init timers
        for (auto& timer : _timers) {
          timer.init();
        }
      }

      void BVH3DCompute::clean()
      {
        for (auto& timer : _timers) {
          timer.destr();
        }
        for (auto& rsrc : _resources) {
          cudaGraphicsUnregisterResource(rsrc);
        }
        for (auto& mrmy : _CUDAMemory) {
          cudaFree(mrmy);
        }
        glDeleteBuffers((GLsizei) _GLbuffers.size(), _GLbuffers.data());
        _iteration = 0;
        _isInitialized = false;
      }

      void BVH3DCompute::compute(unsigned w, unsigned h, unsigned d, unsigned nPos,
                                 GLuint pos_buff, GLuint bounds_buff, Bounds3D bounds)
      {
        // Compute tree size in terms of leaves and nodes
        const uint nLvl = static_cast<uint>(std::ceil(std::log2(nPos))) + 1;
        const uint nLeaf = static_cast<uint>(std::exp2(nLvl - 1));
        const uint nNode = 2u * nLeaf - 1u;

        // NO OpenGL calls beyond this point, mapping CUDA-GL interopability resources
        cudaGraphicsMapResources((int) _resources.size(), _resources.data());

        // Obtain mapped ptrs to interopability resources
        void *pPos;
        void *pNode;
        size_t posSize;
        size_t nodeSize;
        cudaGraphicsResourceGetMappedPointer(&pPos, &posSize, _resources[RSRC_POS_UNSORTED]);
        cudaGraphicsResourceGetMappedPointer(&pNode, &nodeSize, _resources[RSRC_BVH]);

        // Embedding bounds in float4 format
        const float4 maxb = make_float4(bounds.max.x, bounds.max.y, bounds.max.z, 0.f);
        const float4 minb = make_float4(bounds.min.x, bounds.min.y, bounds.min.z, 0.f);
        const float4 range = maxb - minb;
        const float4 invr = make_float4(1.f, 1.f, 1.f, 1.f) / range; // div-by-0 is 'possible'
        
        // Compute morton modes for unsortedembedding positions
        _timers[TIMR_CODES].tick();
        kernComputeMortonCode<<<ceilDiv(nPos, 256u), 256u>>>(
          (float4 *) pPos,
          (uint *) _CUDAMemory[MEMR_CODES_UNSORTED],
          minb, 
          invr, 
          nPos
        );
        _timers[TIMR_CODES].tock();

        // Perform radix sort over embedding positions using morton codes as keys
        _timers[TIMR_SORT].tick();
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          _CUDAMemory[MEMR_SORT_TEMP],
          _memSortTempSize,
          (uint *) _CUDAMemory[MEMR_CODES_UNSORTED], 
          (uint *) _CUDAMemory[MEMR_CODES_SORTED],
          (float4 *) pPos,
          (float4 *) _CUDAMemory[MEMR_POS_SORTED], 
          nPos,
          0,
          29
        );
        _timers[TIMR_SORT].tock();

        // Generate leaf nodes of BVH using sorted embedding positions
        _timers[TIMR_BVH_LEAVES].tick();
        kernComputeBVHLeaf<<<ceilDiv(nLeaf, 256u), 256u>>>(
          (float4 *) _CUDAMemory[MEMR_POS_SORTED],
          (BVHNode *) pNode,
          nLvl - 1,
          nPos,
          nLeaf
        );
        _timers[TIMR_BVH_LEAVES].tock();

        // Generate non-leaf nodes of BVH through a parallel reduction
        _timers[TIMR_BVH_NODES].tick();
        {
          const uint nThreads = 256;
          const uint shMemorySize = (nThreads / 2) * sizeof(BVHNode); 
          const int nLevelsPerIter = static_cast<int>(std::log2(nThreads)) - 1;

          for (int lmax = nLvl - 2; lmax >= 0; lmax -= (nLevelsPerIter + 1)) {
            const int lmin = std::max(lmax - nLevelsPerIter, 0);
            const uint nodesAtPrevLvl = static_cast<uint>(std::pow(2, lmax + 1));
            const uint nWorkgroups = ceilDiv(nodesAtPrevLvl, nThreads);

            kernComputeBVHNode<<<nWorkgroups, nThreads, shMemorySize>>>(
              (BVHNode *) pNode,
              lmax,
              lmin
            );
          }
        }
        _timers[TIMR_BVH_NODES].tock();

        cudaGraphicsUnmapResources((int) _resources.size(), _resources.data());
        // NO OpenGL calls before this point, unmapping CUDA-GL interopability resources
        
        /* // Debug output
        if (_iteration >= _params._iterations - 1) {
          std::vector<BVHNode> buffer(nNode);
          glGetNamedBufferSubData(_GLbuffers[BUFF_BVH], 0, nNode * sizeof(BVHNode), buffer.data()); 

          for (int i = 0; i < 1 + 2 + 4 + 8 + 16 + 32 + 64; i++) {
            std::cout << i << " : " << to_string(buffer[i]) << "\n";
          }
          exit(0);
        } */

        // Update timers
        for (auto& timer : _timers) {
          timer.poll();
        } 

        // Report average times on final iteration
        if (_iteration >= _params._iterations - 1) {
          utils::secureLog(_logger, "BVH computation");
          _timers[TIMR_CODES].log(_logger, "  Morton");
          _timers[TIMR_SORT].log(_logger, "  Radix");
          _timers[TIMR_BVH_LEAVES].log(_logger, "  Leaves");
          _timers[TIMR_BVH_NODES].log(_logger, "  Reduce");
          utils::secureLog(_logger, "");
        }
        _iteration++;
      }
    }
  }
}