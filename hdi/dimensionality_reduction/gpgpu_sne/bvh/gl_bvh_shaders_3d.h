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

#define SHADER_SRC(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

namespace hdi::dr::_3d {
  SHADER_SRC(morton_src, 450,
    struct Bounds {
      vec3 min;
      vec3 max;
      vec3 range;
      vec3 invRange;
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Posi { vec3 positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Boun { Bounds bounds; };
    layout(binding = 2, std430) restrict writeonly buffer Mor { uint morton[]; };
    layout(location = 0) uniform uint nPoints;

    uint expandBits10(uint i)  {
      i = (i * 0x00010001u) & 0xFF0000FFu;
      i = (i * 0x00000101u) & 0x0F00F00Fu;
      i = (i * 0x00000011u) & 0xC30C30C3u;
      i = (i * 0x00000005u) & 0x49249249u;
      return i;
    }

    uint mortonCode(vec3 v) {
      v = clamp(v * 1024.f, 0.f, 1023.f);
      uint x = expandBits10(uint(v.x));
      uint y = expandBits10(uint(v.y));
      uint z = expandBits10(uint(v.z));
      return x * 4u + y * 2u + z;
    }

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      // Normalize position to [0, 1]
      vec3 pos = (positions[i] - bounds.min) * bounds.invRange;
      
      // Compute and store morton code
      morton[i] = mortonCode(pos);
    }
  );

  SHADER_SRC(pos_sorted_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Index { uint indices[]; };
    layout(binding = 1, std430) restrict readonly buffer Posit { vec3 positionsIn[]; };
    layout(binding = 2, std430) restrict writeonly buffer Posi { vec3 positionsOut[]; };
    layout(location = 0) uniform uint nPoints;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      positionsOut[i] = positionsIn[indices[i]];
    }
  );

  SHADER_SRC(leaf_src, 450,
    struct Node {
      vec4 node0; // center of mass (xy/z) and range size (w)
      vec4 node1; // bbox bounds extent (xy/z) and range begin (w)
      vec3 minb;  // bbox minimum bounds
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer MinBB { vec3 minbBuffer[]; };
    layout(binding = 3, std430) restrict readonly buffer Posi { vec3 positionsBuffer[]; };
    layout(binding = 4, std430) restrict readonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 5, std430) restrict readonly buffer Head { uint leafHead; };

    // Uniforms
    layout(location = 0) uniform uint nNodes;

    Node read(uint i) {
      // If node has no mass, return early with empty node
      vec4 node0 = node0Buffer[i];
      if (node0.w == 0) {
        return Node(vec4(0), vec4(0), vec3(0));
      }
      return Node(node0, node1Buffer[i], vec3(0));
    }

    void write(uint i, Node node) {
      node0Buffer[i] = node.node0;
      node1Buffer[i] = node.node1;
      minbBuffer[i] = node.minb;
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

      // Grab data for first embedding point in range
      vec3 pos = positionsBuffer[begin];
      vec3 maxb = pos;
      node.minb = pos;
      node.node0.xyz = pos;

      // Iterate over rest of embedding points in range
      for (uint i = begin + 1; i < end; i++) {
        pos = positionsBuffer[i];
        node.node0.xyz += pos;
        node.minb = min(node.minb, pos);
        maxb = max(maxb, pos);
      }

      node.node0.xyz /= node.node0.w; // Divide center of mass by nr. of embedding points
      node.node1.xyz = maxb - node.minb; // Store extent of bounding box, not maximum

      write(i, node);
    }
  );

  SHADER_SRC(bbox_src, 450,
    struct Node {
      vec4 node0; // center of mass (xy/z) and range size (w)
      vec4 node1; // bbox bounds extent (xy/z) and range begin (w)
      vec3 minb;  // bbox minimum bounds
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer MinBB { vec3 minbBuffer[]; };

    // Uniforms
    layout(location = 0) uniform uint nodeFanout;
    layout(location = 1) uniform uint rangeBegin;
    layout(location = 2) uniform uint rangeEnd;

    // Shared memory
    const uint logk = uint(log2(nodeFanout));
    const uint groupSize = gl_WorkGroupSize.x;
    shared Node sharedNode[groupSize / 2]; // should be smaller for larger fanouts, but "eh"

    Node read(uint i) {
      // If node has no mass, return early with empty node
      vec4 node0 = node0Buffer[i];
      if (node0.w == 0) {
        return Node(vec4(0), vec4(0), vec3(0));
      }
      return Node(node0, node1Buffer[i], minbBuffer[i]);
    }

    void write(uint i, Node node) {
      node0Buffer[i] = node.node0;
      node1Buffer[i] = node.node1;
      minbBuffer[i] = node.minb;
    }

    Node reduce(Node l, Node r) {
      if (r.node0.w == 0) {
        return l;
      } else if (l.node0.w == 0) {
        return r;
      }

      float mass = l.node0.w + r.node0.w;
      float begin = min(l.node1.w, r.node1.w);
      vec3 center = (l.node0.xyz * l.node0.w + r.node0.xyz * r.node0.w)
                   / mass;
      vec3 minb = min(l.minb, r.minb);
      vec3 diam = max(l.node1.xyz + l.minb, r.node1.xyz + r.minb) - minb;

      return Node(vec4(center, mass), vec4(diam, begin), minb);
    }

    void main() {
      const uint i = rangeBegin 
                   + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
      if (i > rangeEnd) {
        return;
      }
      const uint s = gl_LocalInvocationID.x / nodeFanout;
      const uint t = gl_LocalInvocationID.x % nodeFanout;

      // Read in node data per invoc, and let first invoc store in shared memory
      const Node node = read(i);
      if (t == 0) {
        sharedNode[s] = node;
      }
      barrier();

      // Reduce into shared memory over nodeFanout invocs
      for (uint _t = 1; _t < nodeFanout; _t++) {
        if (t == _t && node.node0.w != 0f) {
          sharedNode[s] = reduce(node, sharedNode[s]);
        }
        barrier();
      }

      // Let first invocation store result
      if (t == 0 && sharedNode[s].node0.w > 0) {
        uint j = i / nodeFanout;
        write(j, sharedNode[s]);
      }
    }
  );
}
