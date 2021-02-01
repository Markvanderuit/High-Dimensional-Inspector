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

namespace hdi::dbg::_2d {
  GLSL(field_slice_ext_src, 450,
    // Wrapper structure for bounds data
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    // Wrapper structure for node data
    struct Node {
      vec2 min;
      vec2 max;
      vec2 center; 
      uint begin;
      uint extent;
      vec3 field;
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict readonly buffer VPixe { uvec2 posBuffer[]; };
    layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
    layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
    layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };
    layout(binding = 6, std430) restrict readonly buffer Bound { Bounds bounds; };

    // Image bindings
    layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

    // Uniform values
    layout(location = 0) uniform uint outputLvl;

    Node readNode(uint i) {
      vec4 node0 = node0Buffer[i];
      vec4 node1 = node1Buffer[i];
      node0.xy = (node0.xy - bounds.min) * bounds.invRange;
      node1.xy = node1.xy * bounds.invRange; //= (node1.xy - bounds.min) * bounds.invRange;
      // vec2 _minb = (node0.xy - bounds.min) * bounds.invRange;
      // vec2 _diam = node1.xy * bounds.invRange;
      return Node(
        node0.xy,
        node0.xy + node1.xy,
        node0.xy + 0.5 * node1.xy,
        uint(node1.w),
        uint(node0.w),
        node0.w > 0.f ? fieldBuffer[i].xyz : vec3(0.f)
      );
    }

    // De-interleave 30-wide interleaved bits to 15
    uint shrinkBits15(uint i) {
      i = i & 0x55555555u;               // 1010101010101010101010101010101 (31 bits)
      i = (i | (i >> 1u)) & 0x33333333u; //  110011001100110011001100110011 (30 bits)
      i = (i | (i >> 2u)) & 0x0F0F0F0Fu; //    1111000011110000111100001111 (28 bits)
      i = (i | (i >> 4u)) & 0x00FF00FFu; //        111111110000000011111111 (24 bits)
      i = (i | (i >> 8u)) & 0x0000FFFFu; //                1111111111111111 (16 bits)
      return i;
    }

    // Interleave 15 continuous bits to 30-wide
    uint expandBits15(uint i) {
      i = (i | (i << 8u)) & 0x00FF00FFu; //        111111110000000011111111 (24 bits)
      i = (i | (i << 4u)) & 0x0F0F0F0Fu; //    1111000011110000111100001111 (28 bits)
      i = (i | (i << 2u)) & 0x33333333u; //  110011001100110011001100110011 (30 bits)
      i = (i | (i << 1u)) & 0x55555555u; // 1010101010101010101010101010101 (31 bits)
      return i;
    }

    uint encode(ivec2 uv) {
      uint x = expandBits15(uint(uv.x));
      uint y = expandBits15(uint(uv.y));
      return x | (y << 1);
    }

    ivec2 decode(uint i) {
      uint x = shrinkBits15(i);
      uint y = shrinkBits15(i >> 1);
      return ivec2(x, y);
    }

    uint findLvl(uint i) {
      uint lvl = 0u;
      do { lvl++; } while ((i = ( i - 1) >> BVH_LOGK_2D) > 0);
      return lvl;
    }

    vec3 safeMix(in vec3 a, in vec3 b, in float c, in uint an, in uint bn) {
      if (an > 0 && bn > 0) {
        return mix(a, b, c);
      } else if (an > 0) {
        return a;
      } else if (bn > 0) {
        return b;
      } else {
        return vec3(0);
      }
    }

    void main() {
      // Check if invoc is within nr of items on leaf queue
      const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (j >= leafHead) {
        return;
      }

      // Get info about leaf node
      uint i = leafBuffer[j];
      const Node leafNode = readNode(i);
      
      // Ascend tree towards output level
      uint lvl = findLvl(i);
      while (lvl > outputLvl) {
        lvl--;
        i = (i - 1) >> BVH_LOGK_2D;
      }

      // Determine nr of nodes before output level
      uint nNodes = 0u;
      for (uint i = 0; i < lvl; ++i) {
        nNodes |= 1u << (BVH_LOGK_2D * i);
      }

      // Grab closest representative node, and decode position back to 2d
      const Node nearNode = readNode(i);
      const ivec2 nearPos = decode(i - nNodes);

      // Determine i2n which direction to grab neighbours (left, right, up, down)
      const int xDir = (nearNode.center.x <= leafNode.center.x) ? 1 : -1;
      const int yDir = (nearNode.center.y <= leafNode.center.y) ? 1 : -1;

      // Read neighbour node data. Data is placed along morton order
      const Node horiNode = readNode(nNodes + encode(nearPos + ivec2(xDir, 0)));
      const Node vertNode = readNode(nNodes + encode(nearPos + ivec2(0, yDir)));
      const Node diagNode = readNode(nNodes + encode(nearPos + ivec2(xDir, yDir)));

      // Interpolation factor
      vec2 a = (leafNode.center - nearNode.min) / (nearNode.max - nearNode.min);
      a = abs(a - 0.5f);

      // Bilinear interpolate results,  safeMix() accounts for empty nodes
      vec3 field = safeMix(safeMix(nearNode.field, horiNode.field, a.x, nearNode.extent, horiNode.extent),
                           safeMix(vertNode.field, diagNode.field, a.x, vertNode.extent, diagNode.extent),
                           a.y, nearNode.extent + horiNode.extent, vertNode.extent + diagNode.extent);
      field = nearNode.field;

      // Iterate over pixels
      for (uint k = leafNode.begin; k < leafNode.begin + leafNode.extent; k++) {
        imageStore(fieldImage, ivec2(posBuffer[k]), vec4(field, 0));
      }
    }
  );

  GLSL(field_slice_src, 450, 
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict readonly buffer VPixe { uvec2 posBuffer[]; };
    layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
    layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
    layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };

    // Image bindings
    layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

    // Uniform values
    layout(location = 0) uniform uint outputLvl;

    uint findLvl(uint i) {
      uint lvl = 0u;
      do { lvl++; } while ((i = ( i - 1) >> BVH_LOGK_2D) > 0);
      return lvl;
    }

    void main() {
      // Check if invoc is within nr of items on leaf queue
      const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (j >= leafHead) {
        return;
      }

      const uint i = leafBuffer[j];
      uint _i = i;
      uint lvl = findLvl(_i);

      while (outputLvl < lvl) {
        lvl--;
        _i = (_i - 1) >> BVH_LOGK_2D;
      }

      vec4 field = fieldBuffer[_i];

      // Iterate over pixels
      const uint begin = uint(node1Buffer[i].w);
      const uint extent = uint(node0Buffer[i].w);
      for (uint k = begin; k < begin + extent; k++) {
        imageStore(fieldImage, ivec2(posBuffer[k]), field);
      }
    }
  );

  GLSL(bvh_vert, 450,
    // Wrapper structure for Bounds buffer data
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
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

    layout(location = 0) in vec2 vert; // instanced
    layout(location = 1) in vec4 node0;
    layout(location = 2) in vec4 node1;
    layout(location = 3) in uint flag;
    
    layout(location = 0) out vec2 posOut;
    layout(location = 1) out vec3 colorOut;

    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform float uCubeOpacity;
    layout(location = 2) uniform uint lvl;
    layout(location = 3) uniform bool doLvl;
    layout(location = 4) uniform bool doFlags;

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

    void main() {
      // Calculate range in which instance should fall
      uint nBegin = 0;
      for (uint i = 0; i < lvl; ++i) {
        nBegin |= 1u << (BVH_LOGK_2D * i);
      }
      uint nEnd = nBegin | (1u << (BVH_LOGK_2D * (lvl)));

      // Obtain normalized [0, 1] boundary values
      vec2 _minb = (node0.xy - bounds.min) * bounds.invRange;
      vec2 _diam = node1.xy * bounds.invRange;

      // Generate position from instanced data
      if (node0.w == 0 || (doFlags && flag == 0) || (doLvl && (gl_InstanceID < nBegin || gl_InstanceID >= nEnd))) {
        posOut = vec2(-999); 
      } else {
        posOut = _minb + (0.5 + vert) * _diam;
        posOut.y = 1.f - posOut.y;
      }

      // Output color
      colorOut = labels[doFlags ? 1 : gl_InstanceID % 10] / 255.f;

      // Apply camera transformation to output data
      gl_Position = uTransform * vec4(posOut, 0, 1);
    }
  );

  GLSL(bvh_frag, 450,
    layout(location = 0) in vec2 posIn;
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