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
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_constr.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"

namespace hdi {
  namespace dr {  
    namespace bvh {
      class BVH {
      public:
        BVH();
        ~BVH();

        void init(const TsneParameters &params, GLuint posBuffer, unsigned nPos);
        void destr();
        void compute(const Bounds3D &bounds);

      private:
        // GL Buffer types
        enum BufferType {
          BUFF_BVH,     // Buffer to store entire BVH data: nodes, leaves, points
          BufferTypeLength
        };

        // CUDA-GL interopability graphics resource types
        enum ResourceType {
          RSRC_EMB,     // Resource to read embedding positions to CUDA
          RSRC_BVH,     // Resource to write BVH strucutre to OpenGL
          ResourceTypeLength
        };

        bool _isInit;
        unsigned _iter;
        BVHLayout _layout;
        BVHConstr _constr;
        TsneParameters _params;
        utils::AbstractLog* _logger;
        std::array<GLuint, BufferTypeLength> _buffers;
        std::array<struct cudaGraphicsResource *, ResourceTypeLength> _resources;

      public:
        void setLogger(utils::AbstractLog* logger) {
          _logger = logger; 
          _constr.setLogger(_logger);
        }

        GLuint buffer() const {
          return _buffers[BUFF_BVH];
        }

        BVHLayout layout() const {
          return _layout;
        }
      };
    }
  }
}