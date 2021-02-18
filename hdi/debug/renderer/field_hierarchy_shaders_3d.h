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

namespace hdi::dbg::_3d {
  GLSL(field_slice_src, 450,
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer FFiel { vec4 fieldBuffer[]; };

    // Image bindings
    layout(binding = 0, rgba32f) restrict writeonly uniform image2D outputImage;
    
    // Uniform values
    layout(location = 0) uniform uvec2 dims;
    layout(location = 1) uniform uint lvl;
    layout(location = 2) uniform uint depth;
    layout(location = 3) uniform uint depthDims;

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
      // Check if invoc is within nr of pixels in output texture
      const uvec2 xy = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
      if (min(xy, dims - 1) != xy) {
        return;
      }
      const uvec3 xyz = uvec3(xy, depth);
      
      // Determine nearest neighbouring pixel on hierarchy's level
      const uvec3 px = uvec3(((vec3(xyz) + 0.5) / vec3(dims, depthDims)) * vec3(1u << lvl));

      // Determine address of this pixel in hierarchy
      const uint addr = (0x24924924u >> (32u - BVH_LOGK_3D * lvl)) + encode(px);

      // Sample and proccess field value
      vec4 field = fieldBuffer[addr];
      if (field != vec4(0)) {
        field = vec4(0.5, 0.5, 0.5, 1) + vec4(0.5f * field.yzw, 0);
      } else {
        field = vec4(0.5, 0.5, 0.5, 1);
      }

      // Output field value to image
      imageStore(outputImage, ivec2(xy), field);
    }
  );

  GLSL(bvh_vert, 450,
    // Wrapper structure for Bounds buffer data
    struct Bounds {
      vec3 min;
      vec3 max;
      vec3 range;
      vec3 invRange;
    };
    
    // Ten nice colors for color mapping
    const vec3 labels[10] = vec3[10](
      vec3(16, 78, 139),
      vec3(139, 90, 43),
      vec3(138, 43, 226),
      vec3(0, 128, 0),
      vec3(255, 150, 0),
      vec3(204, 40, 40),
      vec3(131, 139, 131),
      vec3(0, 205, 0),
      vec3(20, 20, 20),
      vec3(0, 150, 255)
    );

    layout(location = 0) in vec3 vert; // instanced
    layout(location = 1) in uint node;
    layout(location = 2) in uint flag;
    
    layout(location = 0) out vec3 posOut;
    layout(location = 1) out vec3 colorOut;

    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform float uCubeOpacity;
    layout(location = 2) uniform uint lvl;
    layout(location = 3) uniform bool doLvl;
    layout(location = 4) uniform bool doFlags;

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

    uint shrinkBits10(uint i) {
      i = i & 0x09249249;
      i = (i | (i >> 2u)) & 0x030C30C3;
      i = (i | (i >> 4u)) & 0x0300F00F;
      i = (i | (i >> 8u)) & 0x030000FF;
      i = (i | (i >> 16u)) & 0x000003FF;
      return i;
    }

    uvec3 decode(uint i) {
      uint x = shrinkBits10(i);
      uint y = shrinkBits10(i >> 1);
      uint z = shrinkBits10(i >> 2);
      return uvec3(x, y, z);
    }
    
    void main() {
      // Determine lvl of node
      uint fLvl = 0;
      {
        uint f = uint(gl_InstanceID);
        do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_3D) > 0);
      }

      // Calculate range in which instance should fall
      const uint offset = (0x24924924u >> (32u - BVH_LOGK_3D * fLvl));
      const uint range = offset + (1u << (BVH_LOGK_3D * fLvl));

      // Calculate bounding box implicitly
      const uvec3 px = decode(uint(gl_InstanceID) - offset);
      const vec3 fbbox = vec3(1) / vec3(1u << fLvl);
      const vec3 fpos = fbbox * (vec3(px) + 0.5);
      
      // Either move vertex out of position, or map to correct space
      if ((node & 3) == 0 
        || (doFlags && flag == 0) 
        || (doLvl && fLvl != lvl)) {
        posOut = vec3(-999); 
      } else {
        posOut = fpos + vert * fbbox;
        // posOut.y = 1.f - posOut.y;
      }

      // Output color
      colorOut = labels[doFlags ? 1 : gl_InstanceID % 10] / 255.f;

      // Apply camera transformation to output data
      gl_Position = uTransform * vec4(posOut, 1);
    }
  );

  GLSL(bvh_frag, 450,
    layout(location = 0) in vec3 posIn;
    layout(location = 1) in vec3 colorIn;

    layout(location = 0) out vec4 colorOut;

    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform float uCubeOpacity;
    layout(location = 2) uniform uint lvl;
    layout(location = 3) uniform bool doLvl;
    layout(location = 4) uniform bool doFlags;
    
    void main() {
      if (uCubeOpacity > 0.f) {
        colorOut =  vec4(colorIn, uCubeOpacity);
      } else {
        discard;
      }
    }
  );
}