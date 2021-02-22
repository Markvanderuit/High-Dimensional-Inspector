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

  // Wrapper structure for bounding box data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec2 ePosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer FNode { uint fNodeBuffer[]; };
  layout(binding = 4, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; };  // Atomic
  layout(binding = 5, std430) restrict readonly buffer IQueu { Pair iQueueBuffer[]; };   // Input
  layout(binding = 6, std430) restrict readonly buffer IQHea { uint iQueueHead; };       // Atomic
  layout(binding = 7, std430) restrict writeonly buffer OQue { Pair oQueueBuffer[]; };   // Output
  layout(binding = 8, std430) restrict readonly buffer OQHea { uint oQueueHead; };       // Atomic
  layout(binding = 9, std430) restrict writeonly buffer LQue { Pair lQueueBuffer[]; };   // Leaf
  layout(binding = 10, std430) restrict coherent buffer LQHe { uint lQueueHead; };       // Atomic
  layout(binding = 11, std430) restrict writeonly buffer DQu { Pair dQueueBuffer[]; };   // Dfs
  layout(binding = 12, std430) restrict coherent buffer DQHe { uint dQueueHead; };       // Atomic
  layout(binding = 13, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
  
  // Uniform values
  layout(location = 0) uniform uint eLvls;    // Nr. of levels in embedding hierarchy
  layout(location = 1) uniform uint fLvls;    // Nr. of levels in field hierarchy
  layout(location = 2) uniform float theta2;  // Squared approximation param. for Barnes-Hut

  // Shared memory
  shared Bounds bounds;

  // Invoc shuffle constants
  const uint thread = gl_LocalInvocationID.x % BVH_KNODE_2D;
  const uint base = gl_LocalInvocationID.x - thread;
  const uint rotate = base + (thread + 1) % BVH_KNODE_2D;

  // Compute dot product of vector with itself, ie. squared eucl. distance to vector from origin
  float sdot(in vec2 v) {
    return dot(v, v);
  }
  
  uint shrinkBits15(uint i) {
    i = i & 0x55555555u;
    i = (i | (i >> 1u)) & 0x33333333u;
    i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
    i = (i | (i >> 4u)) & 0x00FF00FFu;
    i = (i | (i >> 8u)) & 0x0000FFFFu;
    return i;
  }

  uvec2 decode(uint i) {
    uint x = shrinkBits15(i);
    uint y = shrinkBits15(i >> 1);
    return uvec2(x, y);
  }

  void main() {
    const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x)
                     / BVH_KNODE_2D;
    if (idx >= iQueueHead) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Read node pair indices once, share with subgroup operations, then subdivide each side
    Pair pair;
    if (thread == 0) {
      pair = iQueueBuffer[idx];
    }
    pair.f = subgroupShuffle(pair.f, base) * BVH_KNODE_2D + 1 + thread;
    pair.e = subgroupShuffle(pair.e, base) * BVH_KNODE_2D + 1 + thread;

    // Determine node levels based on node pair indices
    uint fLvl = 0;
    uint eLvl = 0;
    {
      uint f = pair.f;
      uint e = pair.e;
      do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_2D) > 0);
      do { eLvl++; } while ((e = (e - 1) >> BVH_LOGK_2D) > 0);
    }

    // Load embedding/field node data. Field node is const, emb node gets shuffled over subgroup
    const uint fNode = fNodeBuffer[pair.f];   // packed as: 30 bits leaf addr, 2 bits node type
    vec4 eNode0 = eNode0Buffer[pair.e];       // packed as: bbox c.o.m., range extent
    vec4 eNode1 = eNode1Buffer[pair.e];       // packed as: bbox extent, range min

    // Determine fNode bbox and center implicitly
    const uvec2 px = decode(pair.f - (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl)));
    const vec2 fbbox = bounds.range / vec2(1u << fLvl);
    const vec2 fpos = bounds.min + fbbox * (vec2(px) + 0.5);

    // Value to accumulate for current field node
    vec3 field = vec3(0);
    
    // Values not rotated around, or used for rotation
    const uint pairBase = pair.e - thread;
    const bool fCanSubdiv = fLvl < fLvls - 1 && (fNode & 3u) > 1u; // ADAPTED HERE

    // Compare node pairs BVH_KNODE_2D times, rotating embedding nodes through the subgroup
    for (uint i = 0; i < BVH_KNODE_2D; i++) {
      // Rotate embedding node index
      pair.e = pairBase + (thread + i) % BVH_KNODE_2D;
      
      if (eNode0.w > 0 && fNode > 0) {
        const bool eCanSubdiv = eLvl < eLvls - 1 && eNode0.w > EMB_BVH_KLEAF_2D;

        // Compute relative squared radii for approximation decision
        // 1. Compute distance between nodes
        const vec2 t = fpos - eNode0.xy;
        const float t2 = dot(t, t); // squared eucl. dist.
        // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
        const vec2 b = normalize(abs(t) * vec2(-1, 1));
        // 3. Compute relative squared diams
        const float r2 = max(
          sdot(eNode1.xy - b * dot(eNode1.xy, b)),
          sdot(fbbox - b * dot(fbbox, b))
        );

        if (r2 / t2 < theta2) {
          // Approximation passes, compute approximated value and truncate
          const float tStud = 1.f / (1.f + t2);
          field += eNode0.w * vec3(tStud, t * (tStud * tStud));
        } else if (fCanSubdiv && eCanSubdiv) {
          // Push pair on work queue for further subdivision
          oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
        /* } else if (eNode0.w > DUAL_BVH_DFS_LEAF && eCanSubdiv) {
          // Leaf is reached in one hierarchy. Optimize subtree traversal in separate shader
          // Skip to bottom of potential singleton chain for leaf nodes
          dQueueBuffer[atomicAdd(dQueueHead, 1)] = Pair(fNode >> 2u, pair.e); */
        // } else {
        } else if (eNode0.w > DUAL_BVH_LARGE_LEAF) {
          // Leaf is reached in both hierarchies. Optimize large leaves in separate shader
          // Skip to bottom of potential singleton chain for leaf nodes
          lQueueBuffer[atomicAdd(lQueueHead, 1)] = Pair(fNode >> 2u, pair.e);
        } else {
          // Leaf is reached in both hierarchies. Compute small leaves directly
          for (uint j = uint(eNode1.w); j < uint(eNode1.w) + uint(eNode0.w); ++j) {
            const vec2 t = fpos - ePosBuffer[j];
            const float tStud = 1.f / (1.f + dot(t, t));
            field += vec3(tStud, t * (tStud * tStud));
          }
        }
      }

      // Shuffle embedding node data over subgroup, rotating it across 4 invocs
      if (i < BVH_KNODE_2D - 1) {
        eNode0 = subgroupShuffle(eNode0, rotate);
        eNode1 = subgroupShuffle(eNode1, rotate);
      }
    }

    // Add computed forces to field hierarchy
    if (field != vec3(0)) {
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
    }
  }
);

GLSL(field_bvh_dual_dfs_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support

  // Wrapper structure for pair queue data
  struct Pair {
    uint f;       // Field hierarchy node index
    uint e;       // Embedding hierarchy node index
  };

  // Wrapper structure for node data
  struct Node {
    vec2 center; 
    uint begin;
    vec2 diam;
    uint extent;
  };

  // Wrapper structure for bounding box data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec2 ePosBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; }; // atomic rw
  layout(binding = 4, std430) restrict readonly buffer DQueu { Pair dQueueBuffer[]; };
  layout(binding = 5, std430) restrict coherent buffer DQHea { uint dQueueHead; }; // atomic rw 
  layout(binding = 6, std430) restrict readonly buffer Bound { Bounds boundsBuffer; };
  
  layout(location = 0) uniform uint eLvls;    // Nr. of levels in embedding hierarchy
  layout(location = 1) uniform uint fLvls;    // Nr. of levels in embedding hierarchy
  layout(location = 2) uniform float theta2;  // Squared approximation param. for Barnes-Hut

  // Shared memory
  shared Bounds bounds;

  // Traversal data
  const uint bitmask = ~((~0u) << BVH_LOGK_2D);
  uint lvl = 0;
  uint loc = 0;
  uint stack = 0;
  vec2 fpos;
  vec2 fbbox;

  uint shrinkBits15(uint i) {
    i = i & 0x55555555u;
    i = (i | (i >> 1u)) & 0x33333333u;
    i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
    i = (i | (i >> 4u)) & 0x00FF00FFu;
    i = (i | (i >> 8u)) & 0x0000FFFFu;
    return i;
  }

  uvec2 decode(uint i) {
    uint x = shrinkBits15(i);
    uint y = shrinkBits15(i >> 1);
    return uvec2(x, y);
  }

  float sdot(in vec2 v) {
    return dot(v, v);
  }

  void descend() {
    // Move down tree
    lvl++;
    loc = loc * BVH_KNODE_2D + 1u;

    // Push unvisited locations on stack
    stack |= (bitmask << (BVH_LOGK_2D * lvl));
  }

  void ascend() {
    // Find distance to next level on stack
    uint nextLvl = findMSB(stack) / BVH_LOGK_2D;
    uint dist = lvl - nextLvl;

    // Move dist up to where next available stack position is
    // and one to the right
    if (dist > 0) {
      loc >>= BVH_LOGK_2D * dist;
    } else {
      loc++;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = BVH_LOGK_2D * lvl;
    uint b = (stack >> shift) - 1;
    stack &= ~(bitmask << shift);
    stack |= b << shift;
  }

  bool approx(inout vec3 field) {
    // Load node data
    Node eNode;
    const vec4 eNode0 = eNode0Buffer[loc];
    if (eNode0.w == 0) {
      return true;
    } else {
      const vec4 eNode1 = eNode1Buffer[loc];
      eNode = Node(eNode0.xy, uint(eNode1.w), eNode1.xy, uint(eNode0.w));
    }

    // Compute relative squared radii for approximation decision
    // 1. Compute distance between nodes
    const vec2 t = fpos - eNode.center;
    const float t2 = dot(t, t);
    // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
    const vec2 b = normalize(abs(t) * vec2(-1, 1));
    // 3. Compute relative squared diameters
    const float r2 = max(
      sdot(eNode.diam - b * dot(eNode.diam, b)),
      sdot(fbbox - b * dot(fbbox, b))
    );

    if (r2 / t2 < theta2) {
      // Approximation passes, compute approximated value and truncate
      const float tStud = 1.f / (1.f + t2);
      field += eNode.extent * vec3(tStud, t * (tStud * tStud));
      return true;
    } else if (eNode.extent <= EMB_BVH_KLEAF_2D || lvl == eLvls - 1) {
      // Iterate over all leaf points (hi, thread divergence here to ruin your day)
      for (uint j = eNode.begin; j < eNode.begin + eNode.extent; ++j) {
        const vec2 t = fpos - ePosBuffer[j];
        const float tStud = 1.f / (1.f +  dot(t, t));
        field += vec3(tStud, t * (tStud * tStud));
      }
      return true;
    }
    return false;
  }

  void main() {
    // Check if invoc is within nr of items on dfs queue
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= dQueueHead) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Load node pair from dfs queue, then load the field hierarchy's leaf
    Pair pair = dQueueBuffer[i];

    // Determine implicit fNode bbox and pos
    const uint fLvl = fLvls - 1;
    const uvec2 px = decode(pair.f - (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl)));
    fbbox = bounds.range / vec2(1u << fLvl);
    fpos = bounds.min + fbbox * (vec2(px) + 0.5);

    // Determine root node level for subtree
    uint root = 0;
    {
      uint e = pair.e;
      do { root++; } while ((e = (e - 1) >> BVH_LOGK_2D) > 0);
    }

    // Set traversal data to start at root's first leftmost child
    lvl = root + 1;
    loc = pair.e * BVH_KNODE_2D + 1u;
    stack = (bitmask << (BVH_LOGK_2D * lvl));

    // Depth-first traverse subtree, accumulating field avlues
    vec3 field = vec3(0);
    do {
      if (approx(field)) {
        ascend();
      } else {
        descend();
      }
    } while (lvl > root);

    // Store data tracked during traversal
    if (field != vec3(0)) {
      // When the field hierarchy is subdivided, each invocation must do its own write
      const uvec3 addr = uvec3(4 * pair.f) + uvec3(0, 1, 2);
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
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
    vec2 center; 
    uint begin;
    vec2 diam;
    uint extent;
  };

  // Wrapper structure for bounding box data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec2 ePosBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; }; // atomic rw
  layout(binding = 4, std430) restrict readonly buffer LQueu { Pair lQueueBuffer[]; };
  layout(binding = 5, std430) restrict coherent buffer LQHea { uint lQueueHead; }; // atomic rw 
  layout(binding = 6, std430) restrict readonly buffer Bound { Bounds boundsBuffer; };
  
  // Uniforms
  layout(location = 0) uniform uint fLvls;    // Nr. of levels in embedding hierarchy

  // Shared memory
  shared Bounds bounds;

  // Constants
  const uint nThreads = 4; // Threads to handle all the buffer reads together
  const uint thread = gl_LocalInvocationID.x % nThreads;

  uint shrinkBits15(uint i) {
    i = i & 0x55555555u;
    i = (i | (i >> 1u)) & 0x33333333u;
    i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
    i = (i | (i >> 4u)) & 0x00FF00FFu;
    i = (i | (i >> 8u)) & 0x0000FFFFu;
    return i;
  }

  uvec2 decode(uint i) {
    uint x = shrinkBits15(i);
    uint y = shrinkBits15(i >> 1);
    return uvec2(x, y);
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    uint j = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
    if (j >= lQueueHead) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Load node pair from queue, skip culled pairs
    Pair pair = lQueueBuffer[j];

    // Determine implicit fNode bbox and pos
    const uint fLvl = fLvls - 1;
    const uvec2 px = decode(pair.f - (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl)));
    vec2 fpos = bounds.min + (bounds.range / vec2(1u << fLvl)) * (vec2(px) + 0.5);
    
    // Read node data using subgroup operations
    const vec4 eNode0 = subgroupShuffle((thread == 0) 
      ? eNode0Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 eNode1 = subgroupShuffle((thread == 2) 
      ? eNode1Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const Node eNode = Node(eNode0.xy, uint(eNode1.w), eNode1.xy, uint(eNode0.w));

    // Directly compute forces for this node pair using group of threads
    vec3 field = vec3(0);
    for (uint i = eNode.begin + thread; 
              i < eNode.begin + eNode.extent; 
              i += nThreads) {
      const vec2 t = fpos - ePosBuffer[i];
      const float tStud = 1.f / (1.f +  dot(t, t));
      field += vec3(tStud, t * (tStud * tStud));
    }
    field = subgroupClusteredAdd(field, nThreads);

    // Add computed forces to field
    if (field != vec3(0) && thread < 3) {
      const uvec3 addr = uvec3(4 * pair.f) + uvec3(0, 1, 2);
      atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
    }
  }
);

GLSL(push_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VPixe { uvec2 pixelsBuffer[]; };
  layout(binding = 1, std430) restrict coherent buffer VFiel { vec3 fieldBuffer[]; };
  
  // Uniforms
  layout(location = 0) uniform uint nPixels;
  layout(location = 1) uniform uint fLvls;
  layout(location = 2) uniform uint startLvl;

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  uint expandBits15(uint i) {
    i = (i | (i << 8u)) & 0x00FF00FFu;
    i = (i | (i << 4u)) & 0x0F0F0F0Fu;
    i = (i | (i << 2u)) & 0x33333333u;
    i = (i | (i << 1u)) & 0x55555555u;
    return i;
  }

  uint encode(uvec2 v) {
    uint x = expandBits15(v.x);
    uint y = expandBits15(v.y);
    return x | (y << 1);
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPixels) {
      return;
    }

    // Read pixel's position
    const uvec2 px = pixelsBuffer[i];

    // Determine address based on pixel position
    // Determine start address based on pixel position
    const uint addr = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (fLvls - 1))) 
                    + encode(px);

    // Determine stop address based on topmost used hierarchy level
    const uint stop = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (startLvl));
        
    // Push forces down by ascending up tree to root
    // ignore root, it will never approximate
    vec3 field = vec3(0);
    for (uint k = addr; k >= stop; k = (k - 1) >> BVH_LOGK_2D) {
      field += fieldBuffer[k];
    }

    // Store resulting value in field image
    imageStore(fieldImage, ivec2(px), vec4(field, 0));
  }
);

/* GLSL(push_smoothed_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Wrapper structure for bounding box data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VPixe { uvec2 pixelsBuffer[]; };
  layout(binding = 1, std430) restrict coherent buffer VFiel { vec3 fieldBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
  
  // Uniforms
  layout(location = 0) uniform uint nPixels;
  layout(location = 1) uniform uint fLvls;
  layout(location = 2) uniform uint startLvl;

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  // Shared memory
  shared Bounds bounds;

  uint expandBits15(uint i) {
    i = (i | (i << 8u)) & 0x00FF00FFu;
    i = (i | (i << 4u)) & 0x0F0F0F0Fu;
    i = (i | (i << 2u)) & 0x33333333u;
    i = (i | (i << 1u)) & 0x55555555u;
    return i;
  }

  uint shrinkBits15(uint i) {
    i = i & 0x55555555u;
    i = (i | (i >> 1u)) & 0x33333333u;
    i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
    i = (i | (i >> 4u)) & 0x00FF00FFu;
    i = (i | (i >> 8u)) & 0x0000FFFFu;
    return i;
  }

  uint encode(uvec2 v) {
    uint x = expandBits15(v.x);
    uint y = expandBits15(v.y);
    return x | (y << 1);
  }

  uvec2 decode(uint i) {
    uint x = shrinkBits15(i);
    uint y = shrinkBits15(i >> 1);
    return uvec2(x, y);
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPixels) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Read pixel's position
    const uvec2 px = pixelsBuffer[i];

    // Sample leaf node for first value
    const uint addr = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (fLvls - 1))) + encode(px);
    vec3 field = fieldBuffer[addr];

    // Compute position of leaf node's center
    const vec2 fbbox = bounds.range / vec2(1u << (fLvls - 1));
    const vec2 fpos = bounds.min + fbbox * (vec2(px) + 0.5f);

    // Step up in tree, sampling four nodes at each turn
    for (uint lvl = fLvls - 2; lvl >= startLvl; lvl--) {
      // Compute position of interpolated pixel given current hierarchy level
      const vec2 fbbox = bounds.range / vec2(1u << lvl);
      const vec2 pxf = ((fpos - bounds.min) / fbbox) - 0.5f;

      // Interpolation factors
      const uvec2 px0 = uvec2(floor(pxf));
      const uvec2 px1 = uvec2(ceil(pxf));
      const vec2 a = pxf - floor(pxf);

      // Sample four nearest nodes on level
      const uint offset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * lvl);
      const vec3 field00 = fieldBuffer[offset + encode(px0)];
      const vec3 field10 = fieldBuffer[offset + encode(uvec2(px1.x, px0.y))];
      const vec3 field01 = fieldBuffer[offset + encode(uvec2(px0.x, px1.y))];
      const vec3 field11 = fieldBuffer[offset + encode(px1)];

      // Interpolate and add
      field += mix(
        mix(field00, field10, a.x),
        mix(field01, field11, a.x),
        a.y
      );
    }

    // Store resulting value in field image
    imageStore(fieldImage, ivec2(px), vec4(field, 0));
  }
); */

/* GLSL(interp_hierarchy_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Wrapper structure for bounding box data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VFie { vec3 fieldBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
  layout(binding = 2, std430) restrict readonly buffer Posi { vec2 posBuffer[]; };
  layout(binding = 3, std430) restrict writeonly buffer Fie { vec3 fieldValues[]; };
  
  // Uniforms
  layout(location = 0) uniform uint nPos;
  layout(location = 1) uniform uint fLvls;
  layout(location = 2) uniform uint startLvl;

  // Shared memory
  shared Bounds bounds;

  uint expandBits15(uint i) {
    i = (i | (i << 8u)) & 0x00FF00FFu;
    i = (i | (i << 4u)) & 0x0F0F0F0Fu;
    i = (i | (i << 2u)) & 0x33333333u;
    i = (i | (i << 1u)) & 0x55555555u;
    return i;
  }

  uint encode(uvec2 v) {
    uint x = expandBits15(v.x);
    uint y = expandBits15(v.y);
    return x | (y << 1);
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPos) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Read query position, transform to [0, 1]
    const vec2 fpos = (posBuffer[i] - bounds.min) * bounds.invRange;

    // Step up in field hierarchy, sampling four closest nodes at each turn
    // and accumulate the different field values
    vec3 field = vec3(0);
    for (uint lvl = fLvls - 1; lvl > startLvl; lvl--) {
      // Transform to continuous pixel position
      const vec2 px = fpos * vec2(1u << lvl) - 0.5f;

      // Determine interpolation factors
      // const vec2 pxfl = floor(px);
      const uvec2 px0 = uvec2(floor(px));
      const uvec2 px1 = uvec2(ceil(px));
      const vec2 a = px - floor(px);

      // Sample four closest nodes on level
      const uint offset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * lvl);
      vec3 field00 = fieldBuffer[offset + encode(px0)];
      vec3 field10 = fieldBuffer[offset + encode(uvec2(px1.x, px0.y))];
      vec3 field01 = fieldBuffer[offset + encode(uvec2(px0.x, px1.y))];
      vec3 field11 = fieldBuffer[offset + encode(px1)];

      // Bilinearly interpolate and add to field value
      // field += field00;
      field += mix(
        mix(field00, field10, a.x),
        mix(field01, field11, a.x),
        a.y
      );
    }

    // Store resulting field value
    fieldValues[i] = field;
    // fieldValues[i] = texture(fieldSampler, fpos).xyz;
  }
); */