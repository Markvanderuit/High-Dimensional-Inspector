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
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec3 ePosBuffer[]; };
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
  const uint thread = gl_LocalInvocationID.x % BVH_KNODE_3D;
  const uint base = gl_LocalInvocationID.x - thread;
  const uint rotate = base + (thread + 1) % BVH_KNODE_3D;

  // Compute dot product of vector with itself, ie. squared eucl. distance to vector from origin
  float sdot(in vec3 v) {
    return dot(v, v);
  }

  uint shrinkBits10(uint i) {
    i = i & 0x09249249;
    i = (i | (i >> 2u)) & 0x030C30C3;
    i = (i | (i >> 4u)) & 0x0300F00F;
    i = (i | (i >> 8u)) & 0x030000FF;
    i = (i | (i >> 16u)) & 0x000003FF;// 0x0000FFFFu; // 11 11 11 11 11 11 11 11
    return i;
  }

  uvec3 decode(uint i) {
    uint x = shrinkBits10(i);
    uint y = shrinkBits10(i >> 1);
    uint z = shrinkBits10(i >> 2);
    return uvec3(x, y, z);
  }

  void main() {
    const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x)
                     / BVH_KNODE_3D;
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
    pair.f = subgroupShuffle(pair.f, base) * BVH_KNODE_3D + 1 + thread;
    pair.e = subgroupShuffle(pair.e, base) * BVH_KNODE_3D + 1 + thread;

    // Determine node levels based on node pair indices
    uint fLvl = 0;
    uint eLvl = 0;
    {
      uint f = pair.f;
      uint e = pair.e;
      do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_3D) > 0);
      do { eLvl++; } while ((e = (e - 1) >> BVH_LOGK_3D) > 0);
    }

    // Load embedding/field node data. Field node is const, emb node gets shuffled over subgroup
    const uint fNode = fNodeBuffer[pair.f];   // packed as: 30 bits leaf addr, 2 bits node type
    vec4 eNode0 = eNode0Buffer[pair.e];       // packed as: bbox c.o.m., range extent
    vec4 eNode1 = eNode1Buffer[pair.e];       // packed as: bbox extent, range min

    // Determine fNode bbox and center implicitly
    const uvec3 px = decode(pair.f - (0x24924924u >> (32u - BVH_LOGK_3D * fLvl)));
    const vec3 fbbox = bounds.range / vec3(1u << fLvl);
    const vec3 fpos = bounds.min + fbbox * (vec3(px) + 0.5);

    // Value to accumulate for current field node
    vec4 field = vec4(0);
    
    // Values not rotated around, or used for rotation
    const uint pairBase = pair.e - thread;
    const bool fCanSubdiv = fLvl < fLvls - 1 && (fNode & 3u) > 1u; // ADAPTED HERE

    // Compare node pairs BVH_KNODE_3D times, rotating embedding nodes through the subgroup
    for (uint i = 0; i < BVH_KNODE_3D; ++i) {
      // Rotate embedding node index
      pair.e = pairBase + (thread + i) % BVH_KNODE_3D;

      if (eNode0.w > 0 && fNode > 0) {
        const bool eCanSubdiv = eLvl < eLvls - 1 && eNode0.w > EMB_BVH_KLEAF_3D;
        
        // Compute relative squared radii for approximation decision
        // 1. Compute distance between nodes
        const vec3 t = fpos - eNode0.xyz;
        const float t2 = dot(t, t); // squared eucl. dist.
        // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
        const vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
        // 3. Compute relative squared diams
        const float r2 = max(
          sdot(eNode1.xyz - b * dot(eNode1.xyz, b)),
          sdot(fbbox - b * dot(fbbox, b))
        );

        if (r2 / t2 < theta2) {
          // Approximation passes, compute approximated value and truncate
          const float tStud = 1.f / (1.f + t2);
          field += eNode0.w * vec4(tStud, t * (tStud * tStud));// / float(fNode.extent);
        } else if (fCanSubdiv && eCanSubdiv) {
          // Push pair on work queue for further subdivision
          oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
        /* } else if (eNode0.w >= DUAL_BVH_DFS_LEAF && eCanSubdiv) {
          // Leaf is reached in one hierarchy. Optimize subtree traversal in separate shader
          // Skip to bottom of potential singleton chain for leaf nodes
          dQueueBuffer[atomicAdd(dQueueHead, 1)] = Pair(fNode >> 2u, pair.e); */
        } else if (eNode0.w > DUAL_BVH_LARGE_LEAF) {
          // Leaf is reached in both hierarchies. Optimize large leaves in separate shader
          // Skip to bottom of potential singleton chain for leaf nodes
          lQueueBuffer[atomicAdd(lQueueHead, 1)] = Pair(fNode >> 2u, pair.e);
        } else {
          // Leaf is reached in both hierarchies. Compute small leaves directly
          for (uint j = uint(eNode1.w); j < uint(eNode1.w) + uint(eNode0.w); ++j) {
            const vec3 t = fpos - ePosBuffer[j];
            const float tStud = 1.f / (1.f +  dot(t, t));
            field += vec4(tStud, t * (tStud * tStud));
          }
        }
      }

      // Shuffle embedding node data over subgroup, rotating it across 4 invocs
      if (i < BVH_KNODE_3D - 1) {
        eNode0 = subgroupShuffle(eNode0, rotate);
        eNode1 = subgroupShuffle(eNode1, rotate);
      }
    }

    // Add computed forces to field hierarchy
    if (field != vec4(0)) {
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
      atomicAdd(fFieldBuffer[addr.w], field.w);
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
    vec3 center; 
    uint begin;
    vec3 diam;
    uint extent;
  };

  // Wrapper structure for bounding box data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec3 ePosBuffer[]; };
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
  const uint bitmask = ~((~0u) << BVH_LOGK_3D);
  uint lvl = 0;
  uint loc = 0;
  uint stack = 0;
  vec3 fpos;
  vec3 fbbox;

  uint shrinkBits10(uint i) {
    i = i & 0x09249249;
    i = (i | (i >> 2u)) & 0x030C30C3;
    i = (i | (i >> 4u)) & 0x0300F00F;
    i = (i | (i >> 8u)) & 0x030000FF;
    i = (i | (i >> 16u)) & 0x000003FF;// 0x0000FFFFu; // 11 11 11 11 11 11 11 11
    return i;
  }

  uvec3 decode(uint i) {
    uint x = shrinkBits10(i);
    uint y = shrinkBits10(i >> 1);
    uint z = shrinkBits10(i >> 2);
    return uvec3(x, y, z);
  }

  float sdot(in vec3 v) {
    return dot(v, v);
  }

  void descend() {
    // Move down tree
    lvl++;
    loc = loc * BVH_KNODE_3D + 1u;

    // Push unvisited locations on stack
    stack |= (bitmask << (BVH_LOGK_3D * lvl));
  }

  void ascend() {
    // Find distance to next level on stack
    uint nextLvl = findMSB(stack) / BVH_LOGK_3D;
    uint dist = lvl - nextLvl;

    // Move dist up to where next available stack position is
    // and one to the right
    if (dist > 0) {
      loc >>= BVH_LOGK_3D * dist;
    } else {
      loc++;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = BVH_LOGK_3D * lvl;
    uint b = (stack >> shift) - 1;
    stack &= ~(bitmask << shift);
    stack |= b << shift;
  }

  bool approx(inout vec4 field) {
    // Load node data
    Node eNode;
    const vec4 eNode0 = eNode0Buffer[loc];
    if (eNode0.w == 0) {
      return true;
    } else {
      const vec4 eNode1 = eNode1Buffer[loc];
      eNode = Node(eNode0.xyz, uint(eNode1.w), eNode1.xyz, uint(eNode0.w));
    }

    // Compute relative squared radii for approximation decision
    // 1. Compute distance between nodes
    const vec3 t = fpos - eNode.center;
    const float t2 = dot(t, t);
    // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
    const vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
    // 3. Compute relative squared diameters
    const float r2 = max(
      sdot(eNode.diam - b * dot(eNode.diam, b)),
      sdot(fbbox - b * dot(fbbox, b))
    );

    if (r2 / t2 < theta2) {
      // Approximation passes, compute approximated value and truncate
      const float tStud = 1.f / (1.f + t2);
      field += eNode.extent * vec4(tStud, t * (tStud * tStud));
      return true;
    } else if (eNode.extent <= EMB_BVH_KLEAF_3D || lvl == eLvls - 1) {
      // Iterate over all leaf points (hi, thread divergence here to ruin your day)
      for (uint j = eNode.begin; j < eNode.begin + eNode.extent; ++j) {
        const vec3 t = fpos - ePosBuffer[j];
        const float tStud = 1.f / (1.f +  dot(t, t));
        field += vec4(tStud, t * (tStud * tStud));
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
    const uvec3 px = decode(pair.f - (0x24924924u >> (32u - BVH_LOGK_3D * fLvl)));
    fbbox = bounds.range / vec3(1u << fLvl);
    fpos = bounds.min + fbbox * (vec3(px) + 0.5);

    // Determine root node level for subtree
    uint root = 0;
    {
      uint e = pair.e;
      do { root++; } while ((e = (e - 1) >> BVH_LOGK_3D) > 0);
    }

    // Set traversal data to start at root's first leftmost child
    lvl = root + 1;
    loc = pair.e * BVH_KNODE_3D + 1u;
    stack = (bitmask << (BVH_LOGK_3D * lvl));

    // Depth-first traverse subtree, accumulating field avlues
    vec4 field = vec4(0);
    do {
      if (approx(field)) {
        ascend();
      } else {
        descend();
      }
    } while (lvl > root);

    // Store data tracked during traversal
    if (field != vec4(0)) {
      // When the field hierarchy is subdivided, each invocation must do its own write
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
      atomicAdd(fFieldBuffer[addr.w], field.w);
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

  // Wrapper structure for bounding box data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer EPosi { vec3 ePosBuffer[]; };
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

  uint shrinkBits10(uint i) {
    i = i & 0x09249249;
    i = (i | (i >> 2u)) & 0x030C30C3;
    i = (i | (i >> 4u)) & 0x0300F00F;
    i = (i | (i >> 8u)) & 0x030000FF;
    i = (i | (i >> 16u)) & 0x000003FF;// 0x0000FFFFu; // 11 11 11 11 11 11 11 11
    return i;
  }

  uvec3 decode(uint i) {
    uint x = shrinkBits10(i);
    uint y = shrinkBits10(i >> 1);
    uint z = shrinkBits10(i >> 2);
    return uvec3(x, y, z);
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
    const uvec3 px = decode(pair.f - (0x24924924u >> (32u - BVH_LOGK_3D * fLvl)));
    vec3 fpos = bounds.min + (bounds.range / vec3(1u << fLvl)) * (vec3(px) + 0.5);
    
    // Read node data using subgroup operations
    const vec4 eNode0 = subgroupShuffle((thread == 0) 
      ? eNode0Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 eNode1 = subgroupShuffle((thread == 2) 
      ? eNode1Buffer[pair.e] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const Node eNode = Node(eNode0.xyz, uint(eNode1.w), eNode1.xyz, uint(eNode0.w));

    // Directly compute forces for this node pair using group of threads
    vec4 field = vec4(0);
    for (uint i = eNode.begin + thread; 
              i < eNode.begin + eNode.extent; 
              i += nThreads) {
      vec3 t = fpos - ePosBuffer[i];
      float tStud = 1.f / (1.f +  dot(t, t));
      field += vec4(tStud, t * (tStud * tStud));
    }
    field = subgroupClusteredAdd(field, nThreads);

    // Add computed forces to field
    if (field != vec4(0) && thread < 4) {
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
    }
  }
);

GLSL(push_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VPixe { uvec3 pixelsBuffer[]; };
  layout(binding = 1, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
  
  // Uniforms
  layout(location = 0) uniform uint nPixels;
  layout(location = 1) uniform uint fLvls;
  layout(location = 2) uniform uint startLvl;

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

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
    // Check if invoc is within nr of items on leaf queue
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPixels) {
      return;
    }

    // Read pixel's position
    const uvec3 px = pixelsBuffer[i];

    // Determine address based on pixel position
    // Determine start address based on pixel position
    const uint addr = (0x24924924u >> (32u - BVH_LOGK_3D * (fLvls - 1))) 
                    + encode(px);
                    
    // Determine stop address based on topmost used hierarchy level
    const uint stop = (0x24924924u >> (32u - BVH_LOGK_3D * (startLvl - 1)))
                    + (1u << (BVH_LOGK_3D * (startLvl - 1)));

    // Push forces down by ascending up tree to root
    // ignore root, it will never approximate
    vec4 field = vec4(0);
    for (uint k = addr; k >= stop; k = (k - 1) >> BVH_LOGK_3D) {
      field += fieldBuffer[k];
    }

    // Store resulting value in field image
    imageStore(fieldImage, ivec3(px), field);
  }
);