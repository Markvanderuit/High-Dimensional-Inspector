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

namespace hdi::dr::_2d {
  GLSL(morton_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Pixe { ivec2 pixels[]; };
    layout(binding = 1, std430) restrict writeonly buffer Mor { uint morton[]; };
    layout(location = 0) uniform uint nPixels;
    layout(location = 1) uniform uvec2 textureSize;

    uint expandBits15(uint i) {
      i = (i | (i << 8u)) & 0x00FF00FFu;
      i = (i | (i << 4u)) & 0x0F0F0F0Fu;
      i = (i | (i << 2u)) & 0x33333333u;
      i = (i | (i << 1u)) & 0x55555555u;
      return i;
    }

    uint mortonCode(vec2 v) {
      v = clamp(v * 32768.f, 0.f, 32767.f);
      uint x = expandBits15(uint(v.x));
      uint y = expandBits15(uint(v.y));
      return x | (y << 1);
    }

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPixels) {
        return;
      }

      // Normalize pixel position to [0, 1]
      vec2 pixel = (vec2(pixels[i]) + 0.5) / vec2(textureSize);
      
      // Compute and store morton code
      morton[i] = mortonCode(pixel);
    }
  );

  GLSL(pixel_sorted_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Index { uint indices[]; };
    layout(binding = 1, std430) restrict readonly buffer Pixel { ivec2 pixelsIn[]; };
    layout(binding = 2, std430) restrict writeonly buffer Pixe { ivec2 pixelsOut[]; };
    layout(location = 0) uniform uint nPixels;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPixels) {
        return;
      }

      pixelsOut[i] = pixelsIn[indices[i]];
    }
  );

  GLSL(divide_dispatch_src, 450,
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer ValueBuffer { uint value; };
    layout(binding = 1, std430) restrict writeonly buffer DispatchBu { uvec3 dispatch; };
    layout(location = 0) uniform uint div;

    void main() {
      dispatch = uvec3((value + div - 1) / div, 1, 1);
    }
  );

  GLSL(subdiv_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer Mort { uint mortonBuffer[]; };
    layout(binding = 1, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 2, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 3, std430) restrict writeonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 4, std430) restrict coherent buffer LHead { uint leafHead; };

    // Uniforms
    layout(location = 0) uniform uint nPixels;
    layout(location = 1) uniform bool isTop;
    layout(location = 2) uniform bool isBottom;
    layout(location = 3) uniform uint rangeBegin;
    layout(location = 4) uniform uint rangeEnd;

    uint findSplit(uint first, uint last) {
      uint firstCode = mortonBuffer[first];
      uint lastCode = mortonBuffer[last];
      uint commonPrefix = findMSB(firstCode ^ lastCode);

      // Initial guess for split position
      uint split = first;
      uint step = last - first;

      // Perform a binary search to find the split position
      do {
        step = (step + 1) >> 1; // Decrease step size
        uint _split = split + step; // Possible new split position

        if (_split < last) {
          uint splitCode = mortonBuffer[_split];
          uint splitPrefix = findMSB(firstCode ^ splitCode);

          // Accept newly proposed split for this iteration
          if (splitPrefix < commonPrefix) {
            split = _split;
          }
        }
      } while (step > 1);

      return split;
    }

    void main() {
      // Check if invoc exceeds range of child nodes we want to compute
      const uint i = rangeBegin 
                   + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) 
                   / BVH_KNODE_2D;
      if (i > rangeEnd) {
        return;
      }
      const uint t = gl_LocalInvocationID.x % BVH_KNODE_2D;

      // Load parent range
      uint mass = uint(node0Buffer[i].w);
      uint begin = uint(node1Buffer[i].w);

      // Override top data in this case
      if (isTop) {
        begin = 0;
        mass = nPixels;
      }

      // Subdivide if the mass is large enough, else truncate
      if (mass > 1) {
        // First, find split position based on morton codes
        // Then set node ranges for left and right child based on split
        // If a range is too small to split, it will be passed to the leftmost invocation only
        uint end = begin + mass - 1;
        for (uint j = BVH_KNODE_2D; j > 1; j /= 2) {
          bool isLeft = (t % j) < (j / 2);
          if (mass > 1) {
            // Node is large enough, split it
            uint split = findSplit(begin, end);
            begin = isLeft ? begin : split + 1;
            end = isLeft ? split : end;
            mass = 1 + end - begin;
          } else {
            // Node is small enough, hand range only to leftmost invocation
            if (!isLeft) {
              begin = 0;
              mass = 0;
              break;
            }
          }
        }
      } else {
        begin = 0;
        mass = 0;
      }

      // Store node data (each invoc stores their own child node)
      uint j = i * BVH_KNODE_2D + 1 + t;
      node0Buffer[j] = vec4(0, 0, 0, mass);
      node1Buffer[j] = vec4(0, 0, 0, begin);

      // Yeet node id on leaf queue if... well if they are a leaf node
      // if (isBottom) {
      // }
      if ((isBottom && mass > 0) || mass == 1) {
        leafBuffer[atomicAdd(leafHead, 1)] = j;
      }
    }
  );

  GLSL(leaf_src, 450,
    struct Node {
      vec4 node0; // Minimum of bbox and range extent
      vec4 node1; // Extent of bbox and range begin
    };
    
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer Field { vec3 fieldBuffer[]; };
    layout(binding = 3, std430) restrict readonly buffer Posi { ivec2 pixelsBuffer[]; };
    layout(binding = 4, std430) restrict readonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 5, std430) restrict readonly buffer Head { uint leafHead; };
    layout(binding = 6, std430) restrict readonly buffer Boun { Bounds bounds; };

    // Uniforms
    layout(location = 0) uniform uint nPixels;
    layout(location = 1) uniform uvec2 textureSize;

    Node read(uint i) {
      vec4 node0 = node0Buffer[i];
      if (node0.w == 0) {
        return Node(vec4(0), vec4(0));
      }
      return Node(node0, node1Buffer[i]);
    }

    void write(uint i, Node node) {
      node0Buffer[i] = node.node0;
      node1Buffer[i] = node.node1;
      fieldBuffer[i] = vec3(0); // Cleared
    }

    void main() {
      // Check if invoc is within nr of items on leaf queue
      const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (j >= leafHead) {
        return;
      }

      // Read node from queue
      const uint i = leafBuffer[j];
      Node node = read(i);

      const uint begin = uint(node.node1.w);
      const uint end = begin + uint(node.node0.w);

      // Size of a leaf's bbox is always one pixel/voxel
      vec2 bbox = bounds.range / vec2(textureSize);

      // Grab data for first pixel in range
      vec2 pos = (vec2(pixelsBuffer[begin]) + 0.5) / vec2(textureSize)
               * bounds.range + bounds.min;
      vec2 maxb = pos + 0.5 * bbox;
      node.node0.xy = pos - 0.5 * bbox;

      // Iterate over rest of pixels in range. 
      // Extremely likely to be skipped, as tree depth is set to match image, so leaf = 1 pixel
      for (uint k = begin + 1; k < end; k++) {
        pos = (vec2(pixelsBuffer[k]) + 0.5) / vec2(textureSize)
            * bounds.range + bounds.min;
        maxb = max(maxb, pos + 0.5 * bbox);
        node.node0.xy = min(node.node0.xy, pos - 0.5 * bbox);
      }
      node.node1.xy = maxb - node.node0.xy; // store extent of bounding box, not maximum

      write(i, node);
    }
  );

  GLSL(bbox_src, 450, 
    struct Node {
      vec4 node0; // Minimum of bbox and range extent
      vec4 node1; // Extent of bbox and range begin
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer Field { vec3 fieldBuffer[]; };

    // Uniforms
    layout(location = 0) uniform uint rangeBegin;
    layout(location = 1) uniform uint rangeEnd;

    // Shared memory
    const uint groupSize = gl_WorkGroupSize.x;
    shared Node sharedNode[groupSize / 2]; // should be smaller for larger fanouts, but "eh"

    Node read(uint i) {
      // If node has no mass, return early with empty node
      vec4 node0 = node0Buffer[i];
      if (node0.w == 0) {
        return Node(vec4(0), vec4(0));
      }
      return Node(node0, node1Buffer[i]);
    }

    void write(uint i, Node node) {
      node0Buffer[i] = node.node0;
      node1Buffer[i] = node.node1;
      fieldBuffer[i] = vec3(0); // Cleared
    }

    Node reduce(Node l, Node r) {
      if (r.node0.w == 0) {
        return l;
      } else if (l.node0.w == 0) {
        return r;
      }

      float mass = l.node0.w + r.node0.w;
      float begin = min(l.node1.w, r.node1.w);
      vec2 minb = min(l.node0.xy, r.node0.xy);
      vec2 maxb = max(l.node1.xy + l.node0.xy, r.node1.xy + r.node0.xy);
      vec2 diam = maxb - minb;

      return Node(vec4(minb, 0, mass), vec4(diam, 0, begin));
    }

    void main() {
      const uint i = rangeBegin + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
      if (i > rangeEnd) {
        return;
      }
      const uint s = gl_LocalInvocationID.x / BVH_KNODE_2D;
      const uint t = gl_LocalInvocationID.x % BVH_KNODE_2D;

      // Read in node data per invoc, and let first invoc store in shared memory
      const Node node = read(i);
      if (t == 0) {
        sharedNode[s] = node;
      }
      barrier();

      // Reduce into shared memory over BVH_KNODE_2D invocs
      for (uint _t = 1; _t < BVH_KNODE_2D; _t++) {
        if (t == _t && node.node0.w != 0f) {
          sharedNode[s] = reduce(node, sharedNode[s]);
        }
        barrier();
      }

      // Let first invocation store result
      if (t == 0 && sharedNode[s].node0.w > 0) {
        uint j = (i - 1) / BVH_KNODE_2D;
        write(j, sharedNode[s]);
      }
    }
  );
}