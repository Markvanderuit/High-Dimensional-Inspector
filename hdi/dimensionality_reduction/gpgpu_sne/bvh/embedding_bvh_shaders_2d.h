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

namespace hdi::dr::_2d {
  SHADER_SRC(morton_src, 450,
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Posi { vec2 positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Boun { Bounds bounds; };
    layout(binding = 2, std430) restrict writeonly buffer Mor { uint morton[]; };
    layout(location = 0) uniform uint nPoints;

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
      if (i >= nPoints) {
        return;
      }

      // Normalize position to [0, 1]
      vec2 pos = (positions[i] - bounds.min) * bounds.invRange;
      
      // Compute and store morton code
      morton[i] = mortonCode(pos);
    }
  );

  SHADER_SRC(pos_sorted_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Index { uint indices[]; };
    layout(binding = 1, std430) restrict readonly buffer Posit { vec2 positionsIn[]; };
    layout(binding = 2, std430) restrict writeonly buffer Posi { vec2 positionsOut[]; };
    layout(location = 0) uniform uint nPoints;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      positionsOut[i] = positionsIn[indices[i]];
    }
  );

   /* SHADER_SRC(subdiv_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer Mort { uint mortonBuffer[]; };
    layout(binding = 1, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 2, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 3, std430) restrict readonly buffer Subd0 { uint subdiv0Buffer[]; };
    layout(binding = 4, std430) restrict readonly buffer SHea0 { uint subdiv0Head; };
    layout(binding = 5, std430) restrict writeonly buffer Sub1 { uint subdiv1Buffer[]; };
    layout(binding = 6, std430) restrict coherent buffer SHea1 { uint subdiv1Head; };
    layout(binding = 7, std430) restrict writeonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 8, std430) restrict coherent buffer LHead { uint leafHead; };

    // Uniforms
    layout(location = 0) uniform uint leafFanout;
    layout(location = 1) uniform uint nodeFanout;
    layout(location = 2) uniform bool isBottom;

    // Constants
    const uint logk = uint(log2(nodeFanout));
    const uint t = gl_LocalInvocationID.x % nodeFanout;

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

      // split = first + (last - first) / 2; // blunt halfway split for testing

      return split;
    }

    void main() {
      // Check if invoc exceeds range of child nodes we want to compute
      const uint _i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nodeFanout;
      if (_i >= subdiv0Head) {
        return;
      }
      const uint i = subdiv0Buffer[_i];

      // Check if invoc exceeds range of child nodes we want to compute
      // const uint i = rangeBegin 
      //              + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) 
      //              / nodeFanout;
      // if (i > rangeEnd) {
      //   return;
      // }

      // Load parent range
      uint begin = uint(node1Buffer[i].w);
      uint mass = uint(node0Buffer[i].w);

      // Subdivide if the mass is large enough
      if (mass <= leafFanout) {
        begin = 0;
        mass = 0;
      } else {
        // First, find split position based on morton codes
        // Then set node ranges for left and right child based on split
        // If a range is too small to split, it will be passed to the leftmost invocation only
        uint end = begin + mass - 1;
        for (uint j = nodeFanout; j > 1; j /= 2) {
          bool isLeft = (t % j) < (j / 2);
          if (mass > leafFanout) {
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
      }

      // Store node data (each invoc stores their own child node)
      uint j = i * nodeFanout + 1 + t;
      node0Buffer[j] = vec4(0, 0, 0, mass);
      node1Buffer[j] = vec4(0, 0, 0, begin);
      
      if (mass > 0) {
        if (isBottom || mass <= leafFanout) {
          // Yeet node id on leaf queue if... well if they are a leaf node
          leafBuffer[atomicAdd(leafHead, 1)] = j;
        } else {
          // Push on subdivision queue for further subdivision
          subdiv1Buffer[atomicAdd(subdiv1Head, 1)] = j;
        }
      }
    }
  ); */

  // Pre-workqueue subdiv
  SHADER_SRC(subdiv_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer Mort { uint mortonBuffer[]; };
    layout(binding = 1, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 2, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 3, std430) restrict writeonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 4, std430) restrict coherent buffer LHead { uint leafHead; };

    // Uniforms
    layout(location = 0) uniform uint leafFanout;
    layout(location = 1) uniform uint nodeFanout;
    layout(location = 2) uniform bool isBottom;
    layout(location = 3) uniform uint rangeBegin;
    layout(location = 4) uniform uint rangeEnd;

    const uint logk = uint(log2(nodeFanout));

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

      // split = first + (last - first) / 2; // blunt halfway split for testing

      return split;
    }

    void main() {
      // Check if invoc exceeds range of child nodes we want to compute
      const uint i = rangeBegin 
                   + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) 
                   / nodeFanout;
      if (i > rangeEnd) {
        return;
      }
      const uint t = gl_LocalInvocationID.x % nodeFanout;

      // Load parent range
      uint begin = uint(node1Buffer[i].w);
      uint mass = uint(node0Buffer[i].w);

      // Subdivide if the mass is large enough
      if (mass <= leafFanout) {
        begin = 0;
        mass = 0;
      } else {
        // First, find split position based on morton codes
        // Then set node ranges for left and right child based on split
        // If a range is too small to split, it will be passed to the leftmost invocation only
        uint end = begin + mass - 1;
        for (uint j = nodeFanout; j > 1; j /= 2) {
          bool isLeft = (t % j) < (j / 2);
          if (mass > leafFanout) {
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
      }

      // Store node data (each invoc stores their own child node)
      uint j = i * nodeFanout + 1 + t;
      node0Buffer[j] = vec4(0, 0, 0, mass);
      node1Buffer[j] = vec4(0, 0, 0, begin);

      // Yeet node id on leaf queue if... well if they are a leaf node
      if (mass > 0 && (isBottom || mass <= leafFanout)) {
        leafBuffer[atomicAdd(leafHead, 1)] = j;
      }
    }
  );

  SHADER_SRC(divide_dispatch_src, 450,
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    struct Dispatch {
      uint num_groups_x;
      uint num_groups_y;
      uint num_groups_z;
    };

    layout(binding = 0, std430) restrict readonly buffer ValueBuffer { uint value; };
    layout(binding = 1, std430) restrict writeonly buffer DispatchBu { Dispatch dispatch; };
    layout(location = 0) uniform uint div;

    void main() {
      dispatch = Dispatch((value + div - 1) / div, 1, 1);
    }
  );

  SHADER_SRC(leaf_src, 450,
    struct Node {
      vec4 node0; // center of mass (xy/z) and range size (w)
      vec4 node1; // bbox bounds extent (xy/z) and range begin (w)
      vec2 minb;  // bbox minimum bounds
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer MinBB { vec2 minbBuffer[]; };
    layout(binding = 3, std430) restrict readonly buffer Posi { vec2 positionsBuffer[]; };
    layout(binding = 4, std430) restrict readonly buffer Leaf { uint leafBuffer[]; };
    layout(binding = 5, std430) restrict readonly buffer Head { uint leafHead; };

    // Uniforms
    layout(location = 0) uniform uint nNodes;

    Node read(uint i) {
      // If node has no mass, return early with empty node
      vec4 node0 = node0Buffer[i];
      if (node0.w == 0) {
        return Node(vec4(0), vec4(0), vec2(0));
      }
      return Node(node0, node1Buffer[i], vec2(0));
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
      vec2 pos = positionsBuffer[begin];
      vec2 maxb = pos;
      node.minb = pos;
      node.node0.xy = pos;

      // Iterate over rest of embedding points in range
      for (uint i = begin + 1; i < end; i++) {
        pos = positionsBuffer[i];
        node.node0.xy += pos;
        node.minb = min(node.minb, pos);
        maxb = max(maxb, pos);
      }

      node.node0.xy /= node.node0.w; // Divide center of mass by nr. of embedding points
      node.node1.xy = maxb - node.minb; // Store extent of bounding box, not maximum

      write(i, node);
    }
  );

  SHADER_SRC(bbox_src, 450,
    struct Node {
      vec4 node0; // center of mass (xy/z) and range size (w)
      vec4 node1; // bbox bounds extent (xy/z) and range begin (w)
      vec2 minb;  // bbox minimum bounds
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
    layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
    layout(binding = 2, std430) restrict buffer MinBB { vec2 minbBuffer[]; };

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
        return Node(vec4(0), vec4(0), vec2(0));
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
      vec2 center = (l.node0.xy * l.node0.w + r.node0.xy * r.node0.w)
                   / mass;
      vec2 minb = min(l.minb, r.minb);
      vec2 diam = max(l.node1.xy + l.minb, r.node1.xy + r.minb) - minb;

      return Node(vec4(center, 0, mass), vec4(diam, 0, begin), minb);
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
} // namespace hdi::dr::_2d


