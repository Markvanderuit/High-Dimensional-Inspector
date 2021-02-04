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

#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

GLSL(field_bvh_dual_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_shuffle : require )   // subgroupShuffle(...) support

  // Wrapper structure for pair queue data
  struct Pair {
    uint f;       // Field hierarchy node index
    uint e;       // Embedding hierarchy node index
  };
  
  // Wrapper structure for hierarchy node
  struct Node {
    vec3 center;  // Center of mass or bounding box
    uint begin;   // Begin of range of contained data
    vec3 diam;    // Diameter of bounding box
    uint extent;  // Extent of range of contained data
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec3 ePosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer FNod0 { vec4 fNode0Buffer[]; };
  layout(binding = 4, std430) restrict readonly buffer FNod1 { vec4 fNode1Buffer[]; };
  layout(binding = 5, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; };  // Atomic
  layout(binding = 6, std430) restrict readonly buffer IQueu { Pair iQueueBuffer[]; };   // Input
  layout(binding = 7, std430) restrict readonly buffer IQHea { uint iQueueHead; };       // Atomic
  layout(binding = 8, std430) restrict writeonly buffer OQue { Pair oQueueBuffer[]; };   // Output
  layout(binding = 9, std430) restrict readonly buffer OQHea { uint oQueueHead; };       // Atomic
  layout(binding = 10, std430) restrict writeonly buffer LQu { Pair lQueueBuffer[]; }; // Leaf
  layout(binding = 11, std430) restrict coherent buffer LQHe { uint lQueueHead; };      // Atomic

  // Uniform values
  layout(location = 0) uniform uint eLvls;    // Nr. of levels in embedding hierarchy
  layout(location = 1) uniform uint fLvls;    // Nr. of levels in field hierarchy
  layout(location = 2) uniform float theta2;  // Squared approximation param. for Barnes-Hut

  // Constants
  const uint thread = gl_LocalInvocationID.x % BVH_KNODE_3D;

  float sdot(in vec3 v) {
    return dot(v, v);
  }

  void main() {
    // Each BVH_KNODE_3D invocations share a single node pair to subdivide
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / BVH_KNODE_3D;
    if (i >= iQueueHead) {
      return;
    }
    
    // Read node pair indices using subgroup operations
    Pair pair;
    if (thread == 0) {
      pair = iQueueBuffer[i];
    }
    pair.f = subgroupShuffle(pair.f, gl_LocalInvocationID.x - thread);
    pair.e = subgroupShuffle(pair.e, gl_LocalInvocationID.x - thread);

    // Read node pair data using subgroup operations
    const vec4 fNode0 = subgroupShuffle((thread == 0) 
      ? fNode0Buffer[pair.f] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 fNode1 = subgroupShuffle((thread == 1) 
      ? fNode1Buffer[pair.f] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    const vec4 eNode0 = subgroupShuffle((thread == 2) 
      ? eNode0Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const vec4 eNode1 = subgroupShuffle((thread == 3) 
      ? eNode1Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 3 - thread);
    Node eNode = Node(eNode0.xyz, uint(eNode1.w), eNode1.xyz, uint(eNode0.w));
    Node fNode = Node(fNode0.xyz + 0.5 * fNode1.xyz, uint(fNode1.w),  fNode1.xyz, uint(fNode0.w));

    // Determine node levels
    uint eLvl = 0;
    uint fLvl = 0;
    {
      uint e = pair.e;
      uint f = pair.f;
      do { eLvl++; } while ((e = (e - 1) >> BVH_LOGK_3D) > 0);
      do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_3D) > 0);
    }

    // Test whether either of these nodes can subdivide (alternatively, one would be a leaf node)
    bool eCanSubdiv = eLvl < eLvls - 1 && eNode.extent > EMB_BVH_KLEAF_3D;
    bool fCanSubdiv = fLvl < fLvls - 1 && fNode.extent > 1;

    // Compute relative squared diameters for subdivision decision
    // 1. Reflect vector between nodes so it always approaches almost-orthogonal to the diameter
    const vec3 b = normalize(abs(fNode.center - eNode.center) * vec3(-1, -1, 1));
    // 2. Compute relative squared diameters
    const float er2 = sdot(eNode.diam - b * dot(eNode.diam, b));
    const float fr2 = sdot(fNode.diam - b * dot(fNode.diam, b));

    // Decision is based on (a) which is a leaf already (b) which has the larger relative diam
    const bool fDidSubdiv = !eCanSubdiv || (fCanSubdiv && fr2 > er2);

    // Perform subdivision on one hierarchy
    if (fDidSubdiv) {
      // Step down in field hierarchy one level, offset thread for fanout
      pair.f = pair.f * BVH_KNODE_3D + 1 + thread;

      // Load new field node
      const vec4 _fNode0 = fNode0Buffer[pair.f]; // bbox minima, range extent
      const vec4 _fNode1 = fNode1Buffer[pair.f]; // bbox extent, range min
      fNode = Node(_fNode0.xyz + 0.5 * _fNode1.xyz, uint(_fNode1.w),  _fNode1.xyz, uint(_fNode0.w));

      // Test if this node can still subdivide (alternatively, it is a leaf node)
      fCanSubdiv = ++fLvl < fLvls - 1 && fNode.extent > 1;
    } else {
      // Step down in embedding hierarchy one level, offset thread for fanout
      pair.e = pair.e * BVH_KNODE_3D + 1 + thread;

      // Load new embedding node
      const vec4 _eNode0 = eNode0Buffer[pair.e]; // bbox center, range extent
      const vec4 _eNode1 = eNode1Buffer[pair.e]; // bbox extent, range min
      eNode = Node(_eNode0.xyz, uint(_eNode1.w), _eNode1.xyz, uint(_eNode0.w));

      // Test if this node can still subdivide (alternatively, it is a leaf node)
      eCanSubdiv = ++eLvl < eLvls - 1 && eNode.extent > EMB_BVH_KLEAF_3D;
    }

    // Test for a dead end in the hierarchy, but keep
    // invocation alive for now
    vec4 field = vec4(0);
    if (eNode.extent != 0 && fNode.extent != 0) {
      // Compute relative squared radii for approximation decision
      // 1. Compute distance between nodes
      const vec3 t = fNode.center - eNode.center;
      const float t2 = dot(t, t);
      // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
      const vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
      // 3. Compute relative squared diameters
      const float r2 = max(
        sdot(eNode.diam - b * dot(eNode.diam, b)),
        sdot(fNode.diam - b * dot(fNode.diam, b))
      );

      // Dual-tree Barnes-Hut approximation is currently based on the
      // formulation in "Accelerating t-SNE using Tree-Based Algorithms", 
      // L. van der Maaten, 2014. Src: http://jmlr.org/papers/v15/vandermaaten14a.html
      // but with a better fitting r parameter computed above. This is essentialy
      // equal to the tangens of a viewing angle of the sphere/bbox/volume
      if (r2 / t2 < theta2) {
        // Approximation passes, compute approximated value and truncate
        const float tStud = 1.f / (1.f + t2);
        field = eNode.extent * vec4(tStud, t * (tStud * tStud));
      } else if (!fCanSubdiv && !eCanSubdiv) {
        // Leaf is reached. Large leaves are dealt with in separate shader
        // as leaves require iterating over all contained data
        if (eNode.extent > DUAL_BVH_LARGE_LEAF) {
          lQueueBuffer[atomicAdd(lQueueHead, 1)] = pair;
        } else {
          for (uint j = eNode.begin; j < eNode.begin + eNode.extent; ++j) {
            const vec3 t = fNode.center - ePosBuffer[j];
            const float tStud = 1.f / (1.f +  dot(t, t));
            field += vec4(tStud, t * (tStud * tStud));
          }
        }
      } else {
        // Push pair on work queue for further subdivision
        oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
      }
    } 

    // Add computed forces to field hierarchy
    if (fDidSubdiv && field != vec4(0)) {
      // When the field hierarchy is subdivided, each invocation must do its own writes
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
      atomicAdd(fFieldBuffer[addr.w], field.w);
    } else if (!fDidSubdiv) {
      // All threads can enter this so the subgroup can be used without inactive invocations
      // When the embedding hierarchy is subdivided, four invocations can write to the same field node
      field = subgroupClusteredAdd(field, BVH_KNODE_3D);
      if (thread < 4 && field != vec4(0)) {
        atomicAdd(fFieldBuffer[4 * pair.f + thread], field[thread]);
      }
    }
  }
);

GLSL(field_bvh_dual_leaf_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_shuffle : require )   // subgroupShuffle(...) support
  
  // Wrapper structure for pair queue data
  struct Pair {
    uint f;       // Field hierarchy node index
    uint e;       // Embedding hierarchy node index
  };

  // Wrapper structure for node data
  struct Node {
    vec3 center; 
    uint begin;
    vec3 diam;
    uint extent;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec3 ePosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer FNod0 { vec4 fNode0Buffer[]; };
  layout(binding = 4, std430) restrict readonly buffer FNod1 { vec4 fNode1Buffer[]; };
  layout(binding = 5, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; }; // atomic rw
  layout(binding = 6, std430) restrict readonly buffer LQueu { Pair lQueueBuffer[]; };
  layout(binding = 7, std430) restrict coherent buffer LQHea { uint lQueueHead; }; // atomic rw 
  
  // Constants
  const uint nThreads = 4; // Two threads to handle all the buffer reads together
  const uint thread = gl_LocalInvocationID.x % nThreads;

  void main() {
    // Check if invoc is within nr of items on leaf queue
    uint j = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
    if (j >= lQueueHead) {
      return;
    }

    // Load node pair from queue, skip culled pairs
    Pair pair = lQueueBuffer[j];

    // Read node data using subgroup operations
    const vec4 fNode0 = subgroupShuffle((thread == 0) 
      ? fNode0Buffer[pair.f] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 fNode1 = subgroupShuffle((thread == 1) 
      ? fNode1Buffer[pair.f] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    const vec4 eNode0 = subgroupShuffle((thread == 2) 
      ? eNode0Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const vec4 eNode1 = subgroupShuffle((thread == 3) 
      ? eNode1Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 3 - thread);
    const Node eNode = Node(eNode0.xyz, uint(eNode1.w), eNode1.xyz, uint(eNode0.w));
    const Node fNode = Node(fNode0.xyz + 0.5 * fNode1.xyz, uint(fNode1.w),  fNode1.xyz, uint(fNode0.w));

    // Compute forces for this node pair directly
    vec4 field = vec4(0);
    for (uint i = eNode.begin + thread; 
              i < eNode.begin + eNode.extent; 
              i += nThreads) {
      vec3 t = fNode.center - ePosBuffer[i];
      float tStud = 1.f / (1.f +  dot(t, t));
      field += vec4(tStud, t * (tStud * tStud));
    }
    field = subgroupClusteredAdd(field, nThreads);

    // Add computed forces to field
    if (field != vec4(0)) {
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
    }
  }
);

/* GLSL(push_ext_src, 450,
  // Wrapper structure for bounds data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  // Wrapper structure for node data
  struct Node {
    vec3 min;
    vec3 center;
    vec3 max;
    uint begin;
    uint extent;
    vec4 field;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer VPixe { uvec3 posBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };
  layout(binding = 6, std430) restrict readonly buffer Bound { Bounds bounds; };

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  Node readNode(uint i) {
    vec4 node0 = node0Buffer[i];
    vec4 node1 = node1Buffer[i];
    node0.xyz = (node0.xyz - bounds.min) * bounds.invRange;
    node1.xyz = node1.xyz * bounds.invRange;
    return Node(
      node0.xyz, 
      node0.xyz + 0.5 * node1.xyz,
      node0.xyz + node1.xyz,
      uint(node1.w),
      uint(node0.w),
      node0.w > 0.f ? fieldBuffer[i] : vec4(0.f)
    );
  }

  // De-interleave 30-wide interleaved bits to 10
  uint shrinkBits10(uint i) {
    i = i & 0x09249249;
    i = (i | (i >> 2u)) & 0x030C30C3;
    i = (i | (i >> 4u)) & 0x0300F00F;
    i = (i | (i >> 8u)) & 0x030000FF;
    i = (i | (i >> 16u)) & 0x000003FF;
    return i;
  }

  // Interleave 10 continuous bits to 30-wide
  // from bit hacks, 4 instructions less
  uint expandBits10(uint i)  {
    i = (i * 0x00010001u) & 0xFF0000FFu;
    i = (i * 0x00000101u) & 0x0F00F00Fu;
    i = (i * 0x00000011u) & 0xC30C30C3u;
    i = (i * 0x00000005u) & 0x49249249u;
    return i;
  }

  uint encode(ivec3 uv) {
    uint x = expandBits15(uint(uv.x));
    uint y = expandBits15(uint(uv.y));
    uint z = expandBits15(uint(uv.z));
    return x | (y << 1) | (z << 2);
  }

  ivec3 decode(uint i) {
    uint x = shrinkBits15(i);
    uint y = shrinkBits15(i >> 1);
    uint z = shrinkBits15(i >> 2);
    return ivec3(x, y, z);
  }

  uint findLvl(uint i) {
    uint lvl = 0u;
    do { lvl++; } while ((i = ( i - 1) >> BVH_LOGK_2D) > 0);
    return lvl;
  }

  vec4 safeMix(in vec4 a, in vec4 b, in float c, in uint an, in uint bn) {
    if (an > 0 && bn > 0) {
      return mix(a, b, c);
    } else if (an > 0) {
      return a;
    } else if (bn > 0) {
      return b;
    } else {
      return vec4(0);
    }
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (j >= leafHead) {
      return;
    }

    // Get leaf node
    uint i = leafBuffer[j];
    const Node leafNode = readNode(i);

    // Determine nr of nodes before current level
    uint lvl = findLvl(i);
    uint nNodes = 0u;
    for (uint i = 0; i < lvl; ++i) {
      nNodes |= 1u << (BVH_LOGK_2D * i);
    }

    vec4 field = leafNode.field;
    while (lvl > 0) { // 0 will never approximate, so it can be skipped
      // Ascend one level in hierarchy
      lvl--;
      i = (i - 1) >> BVH_LOGK_2D;
      nNodes -= 1u << (BVH_LOGK_2D * lvl); // FIXME check on paper

      // Get new node
      const Node nearNode = readNode(i);
      const ivec2 nearPos = decode(i - nNodes);

      // Determine in which direction to grab neighbours (left, right, up, down)
      const int xDir = (nearNode.center.x <= leafNode.center.x) ? 1 : -1;
      const int yDir = (nearNode.center.y <= leafNode.center.y) ? 1 : -1;
      const int zDir = (nearNode.center.z <= leafNode.center.z) ? 1 : -1;

      // Read neighbour node data. Data is placed along morton order
      const Node horiNode = readNode(nNodes + encode(nearPos + ivec2(xDir, 0)));
      const Node vertNode = readNode(nNodes + encode(nearPos + ivec2(0, yDir)));
      const Node diagNode = readNode(nNodes + encode(nearPos + ivec2(xDir, yDir)));

      // Interpolation factor
      vec2 a = (leafNode.center - nearNode.min) / (nearNode.max - nearNode.min);
      a = abs(a - 0.5f);

      // Interpolate and add to leaf node
      field += safeMix(safeMix(nearNode.field, horiNode.field, a.x, nearNode.extent, horiNode.extent),
                       safeMix(vertNode.field, diagNode.field, a.x, vertNode.extent, diagNode.extent),
                       a.y, nearNode.extent + horiNode.extent, vertNode.extent + diagNode.extent);
      // field += nearNode.field;
    }

    for (uint i = leafNode.begin; i < leafNode.begin + leafNode.extent; i++) {
      imageStore(fieldImage, ivec2(posBuffer[i]), vec4(field, 0));
    }
  }
); */

GLSL(push_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer VPixe { uvec3 posBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (j >= leafHead) {
      return;
    }

    const uint i = leafBuffer[j];
    
    // Push forces down by ascending up tree to root
    vec4 field = vec4(0); // fieldBuffer[0]; // ignore root, it will never approximate
    for (int k = int(i); k > 0; k = (k - 1) >> BVH_LOGK_3D) {
      field += fieldBuffer[k];
    }

    // Store resulting field value in underlying image pixel(s)
    const uint begin = uint(node1Buffer[i].w);
    const uint extent = uint(node0Buffer[i].w);
    for (uint k = begin; k < begin + extent; k++) {
      imageStore(fieldImage, ivec3(posBuffer[k]), field);
    }
  }
);