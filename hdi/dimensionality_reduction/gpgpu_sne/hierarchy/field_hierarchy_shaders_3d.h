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

#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

namespace hdi::dr::_3d {
  GLSL(field_hierarchy_leaf_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict writeonly buffer Nod { uint nodeBuffer[]; };
    layout(binding = 1, std430) restrict readonly buffer Pixe { uvec3 pixelsBuffer[]; };

    // Uniforms
    layout(location = 0) uniform uint nPixels;
    layout(location = 1) uniform uint lvl;

    uint expandBits10(uint i) {
      i = (i | (i << 16u)) & 0x030000FF;
      i = (i | (i <<  8u)) & 0x0300F00F;
      i = (i | (i <<  4u)) & 0x030C30C3;
      i = (i | (i <<  2u)) & 0x09249249;
      return i;
    }

    uint encode(uvec3 v) {
      uint x = expandBits10(v.x);
      uint y = expandBits10(v.y);
      uint z = expandBits10(v.z);
      return x | (y << 1) | (z << 2);
    }

    void main() {
      // Check if invoc is within nr of selected pixels
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPixels) {
        return;
      }

      // Read pixel coordinates
      const uvec3 px = pixelsBuffer[i];

      // Address of pixel in hierarchy's leaf level
      const uint offset = (0x24924924u >> (32u - BVH_LOGK_3D * lvl));
      const uint j = offset + encode(px);

      // Store node as leaf with address to self encoded within 30 msb
      nodeBuffer[j] = (j << 2u) | 1u;
    }
  );

  GLSL(field_hierarchy_reduce_src, 450,
    GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
   
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Nod { uint nodeBuffer[]; };

    // Uniforms
    layout(location = 0) uniform uint lvl;
    
    void main() {
      // Offset and range of addresses for current level
      const uint offset = (0x24924924u >> (32u - BVH_LOGK_3D * lvl));
      const uint range = offset + (1u << (BVH_LOGK_3D * lvl));

      // Check if invoc is in range of addresses
      const uint i = offset + gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= range) {
        return;
      }

      // Read in node mass per child and reduce to a single parent node
      const uint child = nodeBuffer[i];
      
      // Least significant 2 bits form type, remaining 30 form leaf address to skip singleton chains
      const uint type = min(3u, subgroupClusteredAdd(child & 3u, BVH_KNODE_3D)); // extract and add together types
      const uint j = (type > 1u) ? 0u : subgroupClusteredAnd(child >> 2u, BVH_KNODE_3D); 

      // Write parent node
      if (gl_LocalInvocationID.x % BVH_KNODE_3D == 0) {
        nodeBuffer[(i - 1) >> BVH_LOGK_3D] = (j << 2u) | type;
      }
    }
  );
} //namespace hdi::dr::_3d 