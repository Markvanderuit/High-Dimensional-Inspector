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
  layout(binding = 11, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
  layout(binding = 12, std430) restrict writeonly buffer RQu { Pair rQueueBuffer[]; };       // Atomic
  layout(binding = 13, std430) restrict coherent buffer RQHe { uint rQueueHead; };       // Atomic
  
  // Uniform values
  layout(location = 0) uniform uint dhLvl;    // Current level in virtual dual hierarchy
  layout(location = 1) uniform uint eLvls;    // Total nr. of levels in embedding hierarchy
  layout(location = 2) uniform uint fLvls;    // Total nr. of levels in field hierarchy
  layout(location = 3) uniform float theta2;  // Squared approximation param. for Barnes-Hut
  layout(location = 4) uniform bool divideFurther;

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

    // Load embedding/field node data. Field node is const, emb node gets shuffled over subgroup
    const uint fNode = fNodeBuffer[pair.f];   // packed as: 30 bits leaf addr, 2 bits node type
    vec4 eNode0 = eNode0Buffer[pair.e];       // packed as: bbox c.o.m., range extent
    vec4 eNode1 = eNode1Buffer[pair.e];       // packed as: bbox extent, range min

    // Move down singleton chain to leaf node if necessary
    if ((fNode & 3u) == 1u) {
      pair.f = fNode >> 2u;
    }

    // Determine fNode bbox and center implicitly
    const uint fLvl = (fNode & 3u) == 1u ? fLvls - 1 : dhLvl;
    const uvec3 px = decode(pair.f - (0x24924924u >> (32u - BVH_LOGK_3D * fLvl)));
    const vec3 fbbox = bounds.range / vec3(1u << fLvl);
    const vec3 fpos = bounds.min + fbbox * (vec3(px) + 0.5);

    // Values accumulated or kept constant during rotation
    const uint pairBase = pair.e - thread;
    const bool fIsLeaf = (fNode & 3u) == 1u;
    const uint eOffset = 0x24924924u >> (32u - BVH_LOGK_3D * (eLvls - 1));
    vec4 field = vec4(0);    

    // Compare node pairs BVH_KNODE_3D times, rotating embedding nodes through the subgroup
    for (uint i = 0; i < BVH_KNODE_3D; ++i) {
      // Rotate embedding node data over BVH_KNODE_3D invocs in the subgroup
      eNode0 = subgroupShuffle(eNode0, rotate);
      eNode1 = subgroupShuffle(eNode1, rotate);

      // Prevent computation for dead node pairs. Yay divergence
      if (eNode0.w == 0 || fNode == 0) {
        continue;
      }

      // Barnes-Hut approximation components
      // 1. Compute vector dist. and squared eucl dist. between nodes
      const vec3 t = fpos - eNode0.xyz;
      const float t2 = dot(t, t);
      // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
      const vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
      // 3. Compute relative squared diams
      const float r2 = max(
        sdot(eNode1.xyz - b * dot(eNode1.xyz, b)),
        sdot(fbbox - b * dot(fbbox, b))
      );
      
      // More rotate data
      const uint rot_e = pairBase + (thread + i + 1) % BVH_KNODE_3D;
      const bool eIsLeaf = eNode0.w <= EMB_BVH_KLEAF_3D || rot_e >= eOffset;
      
      // Barnes-Hut test
      if (r2 / t2 < theta2) {
        // Approximation passes, compute approximated value and truncate
        const float tStud = 1.f / (1.f + t2);
        field += eNode0.w * vec4(tStud, t * (tStud * tStud));
      } else if (!eIsLeaf && !fIsLeaf) {
        // Push pair on work queue for further subdivision
        oQueueBuffer[atomicAdd(oQueueHead, 1)] = Pair(pair.f, rot_e);
      /* } else if ((!eIsLeaf || !fIsLeaf) && divideFurther) {
        // Push pair on secondary work queue for further subdivision
        rQueueBuffer[atomicAdd(rQueueHead, 1)] = Pair(pair.f, rot_e); */
      } else if (eNode0.w > DUAL_BVH_LARGE_LEAF) {
        // Optimize large leaf pairs in separate shader
        lQueueBuffer[atomicAdd(lQueueHead, 1)] = Pair(pair.f, rot_e);
      } else {
        // Compute small leaves directly
        const uint begin = uint(eNode1.w);
        const uint end = begin + uint(eNode0.w);
        for (uint j = begin; j < end; ++j) {
          const vec3 t = fpos - ePosBuffer[j];
          const float tStud = 1.f / (1.f +  dot(t, t));
          field += vec4(tStud, t * (tStud * tStud));
        }
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

GLSL(field_bvh_dual_rest_src, 450,
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
  layout(binding = 11, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };
  
  // Uniform values
  layout(location = 0) uniform uint eLvls;    // Total nr. of levels in embedding hierarchy
  layout(location = 1) uniform uint fLvls;    // Total nr. of levels in field hierarchy
  layout(location = 2) uniform float theta2;  // Squared approximation param. for Barnes-Hut
  
  // Shared memory
  shared Bounds bounds;

  // Constants
  const uint nThreads = BVH_KNODE_3D;
  const uint thread = gl_SubgroupInvocationID % nThreads;
  const uint base = gl_SubgroupInvocationID - thread;
  const uint fLeafOffset = 0x24924924u >> (32u - BVH_LOGK_3D * (fLvls - 1));
  const uint eLeafOffset = 0x24924924u >> (32u - BVH_LOGK_3D * (eLvls - 1));

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
    const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / BVH_KNODE_3D;
    if (idx >= iQueueHead) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Read node pair indices once, share with subgroup operations
    Pair pair;
    if (thread == 0) {
      pair = iQueueBuffer[idx];
    }
    pair.f = subgroupShuffle(pair.f, base);
    pair.e = subgroupShuffle(pair.e, base);

    // Determine node pair's levels in hierarchies
    uint fLvl = 0;
    uint eLvl = 0;
    {
      uint f = pair.f;
      uint e = pair.e;
      do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_3D) > 0);
      do { eLvl++; } while ((e = (e - 1) >> BVH_LOGK_3D) > 0);
    }

    // Read node data for determining subdivision
    uint fNode; // loaded at a later point
    vec4 eNode0 = subgroupShuffle((thread == 0) ? eNode0Buffer[pair.e] : vec4(0), base);
    vec4 eNode1 = subgroupShuffle((thread == 2) ? eNode1Buffer[pair.e] : vec4(0), base + 2);

    // Determine whether we are dealing with leaf nodes
    bool fIsLeaf = (pair.f >= fLeafOffset);
    bool eIsLeaf = (pair.e >= eLeafOffset) || (eNode0.w <= EMB_BVH_KLEAF_3D);    

    // Subdivide one side. All invocs subdivide the same side!
    // TODO subdivide the larger projected diagonal node
    const bool eSubdiv = fIsLeaf || (!eIsLeaf && eLvl < fLvl);
    if (eSubdiv) {
      // Descend hierarchy
      eLvl++;
      pair.e = pair.e * BVH_KNODE_3D + 1 + thread;
      
      // Load new node data (now we can load fNode)
      fNode = subgroupShuffle((thread == 0) ? fNodeBuffer[pair.f] : 0u, base);
      eNode0 = eNode0Buffer[pair.e];
      eNode1 = eNode1Buffer[pair.e];

      // Update leaf status
      eIsLeaf = (pair.e >= eLeafOffset) || (eNode0.w <= EMB_BVH_KLEAF_3D);
    } else {
      // Descend hierarchy
      fLvl++;
      pair.f = pair.f * BVH_KNODE_3D + 1 + thread;

      // Load new node data (other node already loaded)
      fNode = fNodeBuffer[pair.f];

      // Update leaf status
      fIsLeaf = (pair.f >= fLeafOffset) || ((fNode & 3u) == 1u);
    }

    // Skip singleton chain in field hierarchy if encountered
    if ((fNode & 3u) == 1u) {
      pair.f = fNode >> 2u;
      fLvl = fLvls - 1;
      fIsLeaf = true;
    }

    // Skip computation if dead node is encountered
    vec4 field = vec4(0);
    if (eNode0.w != 0 && fNode != 0) {
      // Determine fNode bbox and center implicitly
      const uvec3 px = decode(pair.f - (0x24924924u >> (32u - BVH_LOGK_3D * fLvl)));
      const vec3 fbbox = bounds.range / vec3(1u << fLvl);
      const vec3 fpos = bounds.min + fbbox * (vec3(px) + 0.5);

      // Barnes-Hut approximation components
      // 1. Compute vector dist. and squared eucl dist. between nodes
      const vec3 t = fpos - eNode0.xyz;
      const float t2 = dot(t, t);
      // 2. Reflect vector so it always approaches almost-orthogonal to the diameter
      const vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
      // 3. Compute relative squared diams
      const float r2 = max(
        sdot(eNode1.xyz - b * dot(eNode1.xyz, b)),
        sdot(fbbox - b * dot(fbbox, b))
      );

      // Barnes-Hut test
      if (r2 / t2 < theta2) {
        // Approximation passes, compute approximated value and truncate
        const float tStud = 1.f / (1.f + t2);
        field += eNode0.w * vec4(tStud, t * (tStud * tStud));
      } else if (!eIsLeaf || !fIsLeaf) {
        // Push pair on work queue for further subdivision
        oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
      } else if (eNode0.w >= DUAL_BVH_LARGE_LEAF) {
        // Optimize large leaf pairs in separate shader
        lQueueBuffer[atomicAdd(lQueueHead, 1)] = pair;
      } else {
        // Compute small leaves directly
        const uint begin = uint(eNode1.w);
        const uint end = begin + uint(eNode0.w);
        for (uint j = begin; j < end; ++j) {
          const vec3 t = fpos - ePosBuffer[j];
          const float tStud = 1.f / (1.f +  dot(t, t));
          field += vec4(tStud, t * (tStud * tStud));
        }
      }
    }
    
    if (eSubdiv) {
      field = subgroupClusteredAdd(field, nThreads);
    }

    if (field != vec4(0)) {
      const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
      if (!eSubdiv) {
        atomicAdd(fFieldBuffer[addr.x], field.x);
        atomicAdd(fFieldBuffer[addr.y], field.y);
        atomicAdd(fFieldBuffer[addr.z], field.z);
        atomicAdd(fFieldBuffer[addr.w], field.w);
      } else if (thread < 4) {
        // Long live subgroups!
        atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
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
  const uint nThreads = 4; // Nr. of threads to handle all the buffer reads together
  const uint thread = gl_SubgroupInvocationID % nThreads;
  const uint base = gl_SubgroupInvocationID - thread;
  const uint foffset = 0x24924924u >> (32u - BVH_LOGK_3D * (fLvls - 1));
  const vec3 fbbox = vec3(1.f / float(1u << (fLvls - 1)));

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
    const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
    if (idx >= lQueueHead) {
      return;
    }

    // Load bounds buffer into shared memory
    if (gl_LocalInvocationID.x == 0) {
      bounds = boundsBuffer;
    }
    barrier();

    // Load node pair from queue using subgroup operations
    Pair pair;
    if (thread == 0) {
      pair = lQueueBuffer[idx];
    }
    pair.f = subgroupShuffle(pair.f, base);
    pair.e = subgroupShuffle(pair.e, base);
    
    // Read node data using subgroup operations
    const vec4 eNode0 = subgroupShuffle((thread == 0) ? eNode0Buffer[pair.e] : vec4(0), base);
    const vec4 eNode1 = subgroupShuffle((thread == 2) ? eNode1Buffer[pair.e] : vec4(0), base + 2);

    // Determine field node position
    const vec3 fpos = bounds.min + bounds.range * fbbox * (vec3(decode(pair.f - foffset)) + 0.5);

    // Range of embedding positions to iterate over
    const uint eBegin = uint(eNode1.w) + thread;
    const uint eEnd = uint(eNode1.w) + uint(eNode0.w);

    // Directly compute forces for this node pair using group of threads
    vec4 field = vec4(0);
    for (uint i = eBegin; i < eEnd; i += nThreads) {
      const vec3 t = fpos - ePosBuffer[i];
      const float tStud = 1.f / (1.f +  dot(t, t));
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

  // Constants
  const uint offset = 0x24924924u >> (32u - BVH_LOGK_3D * (fLvls - 1));
  const uint stop = 0x24924924u >> (32u - BVH_LOGK_3D * (startLvl));

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

    // Push forces down by ascending up tree to root
    // ignore root, it will never approximate
    vec4 field = vec4(0);
    for (uint k = offset + encode(px);
         k >= stop;
         k = (k - 1) >> BVH_LOGK_3D) {
      field += fieldBuffer[k];
    }

    // Store resulting value in field image
    imageStore(fieldImage, ivec3(px), field);
  }
);