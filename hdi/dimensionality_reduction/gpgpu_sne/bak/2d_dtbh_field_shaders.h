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
#include "2d_bh_field_shaders.h"

/**
 * Dual-tree traversal root shader.
 * 
 * Performs field computation using a barnes-hut dual tree approximation, but only
 * considers the different subtrees' root nodes. Remaining nodes requiring subdivision 
 * are traversed entirely in the next shader invocation. Root nodes are dealt with 
 * separately because we leverage multiple invocations for comparing an entire level 
 * of non-root nodes, but cannot do so on the top level.
 * 
 * @author Mark van de Ruit
 */
GLSL(dt_root_src, 450,
  #extension GL_NV_shader_atomic_float : require\n // imageAtomicAdd(f32) support

  // Wrapper structure for AtlasBuffer data
  struct AtlasLevel {
    uvec2 offset;
    uvec2 size;
  };

  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  // Wrapper structure for RootsBuffer data
  struct Roots {
    uint lidx;
    uint ridx;
    bool doSubdivide;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 1, std430) restrict readonly buffer AtlasBuffer { AtlasLevel atlasLevels[]; };
  layout(binding = 2, std430) restrict buffer RootsBuffer { Roots roots[]; };
  layout(binding = 0, r32f) coherent uniform image2DArray fieldImage;
  layout(location = 0) uniform uint nComps;
  layout(location = 1) uniform uint nArrayLevels;
  layout(location = 2) uniform uint topLevel;
  layout(location = 3) uniform float theta2;
  layout(location = 4) uniform sampler2D treeAtlasSampler;

  // Shared memory for texture atlas layout
  shared ivec2 shAtlasDims;
  shared ivec2 shAtlasOffset;
  shared float shAtlasDiams2;

  // Threads operate on a different image array level to reduce resource contention
  const uint arrayAdd = 3u * (gl_WorkGroupID.x % nArrayLevels); 

  /**
   * Atomic image addition to field images.
   * 
   * Splits provided components and atomically adds each to a separate bound image. This
   * is necessary as imageAtomicAdd(f32) does not come with multi-channel support.
   */
  void fieldAtomicAdd(in ivec2 loc, in vec3 v) {
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 0), v.x);
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 1), v.y);
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 2), v.z);
  }

  void main() {
    // Invocation ID
    const uint lid = gl_LocalInvocationID.x;
    const uint gid = gl_WorkGroupID.x 
                   * gl_WorkGroupSize.x 
                   + gl_LocalInvocationID.x;

    // Read top level atlas layout into shared memory
    // Store squared diameter of tree node for BH approximation test
    if (lid < 1) {
      const AtlasLevel lvl = atlasLevels[topLevel];
      shAtlasOffset = ivec2(lvl.offset);
      shAtlasDims = ivec2(lvl.size);
      shAtlasDiams2 = pow(length(vec2(bounds.range) / vec2(lvl.size)), 2);
    }
    barrier();

    // Valid tree nodes given the current invocation ID?
    if (gid >= nComps) {
      return;
    }

    // Get 1d indices of left and right root nodes we will compare
    // We precomputed these because of floating point inaccuracies in sqrt()
    const Roots root = roots[gid];

    // Start traversal at tree root nodes
    // Add offsets into tree texture atlas
    const ivec2 lloc = shAtlasOffset 
      + ivec2(root.lidx % shAtlasDims.x, root.lidx / shAtlasDims.x);
    const ivec2 rloc = shAtlasOffset 
      + ivec2(root.ridx % shAtlasDims.x, root.ridx / shAtlasDims.x);

    // Sample textures at left, right root nodes
    const vec4 lnode = texelFetch(treeAtlasSampler, lloc, 0);
    const vec4 rnode = texelFetch(treeAtlasSampler, rloc, 0);
    
    // Squared eucl. distance between nodes
    const vec2 t = lnode.yz - rnode.yz;
    const float t2 = dot(t, t);

    // If at least one node is empty, we truncate
    const bool isEmpty = lnode.x == 0.f || rnode.x == 0.f;

    // If node pair forms a suitable approximation, we truncate
    const bool isApprox = (lnode.x == 1.f && rnode.x == 1.f)
                      || (shAtlasDiams2 / t2) < theta2;
    
    // Flag for subdivision in next shader pass
    roots[gid].doSubdivide = !isEmpty && !isApprox;

    if (!isEmpty && isApprox && (lloc != rloc)) {
      // Compute field values
      const float studt = 1.f / (1.f + t2);
      const vec2 studt2 = t * (studt * studt);

      // Add points x forces to both trees if they are different
      fieldAtomicAdd(lloc, rnode.x * vec3(studt, studt2));
      fieldAtomicAdd(rloc, lnode.x * vec3(studt, -studt2));
    }
  }
);

/**
 * Dual-tree traversal shader.
 * 
 * Performs field computation using a barnes-hut dual tree approximation. Field
 * values are stored on different locations of an atlas texture, depending on
 * the tree level in which they were computed. These are added together in
 * a later shader.
 * 
 * @author Mark van de Ruit
 */
GLSL(dt_tree_src, 450,
  #extension GL_NV_shader_atomic_float : require\n // imageAtomicAdd(f32) support
  #extension GL_KHR_shader_subgroup_shuffle : require\n // subgroupShuffle() support
  #extension GL_KHR_shader_subgroup_clustered : require\n // subgroupClusteredOr() support
  
  // Queue size defined by nr of levels we traverse
  // We pack the queue tightly, as two levels fit into a single uint32_t
  const uint nLevels = 8u;
  const uint queueSize = nLevels / 2u;

  // Nr. of and IDs of invocations in a subgroup handling a single dual-tree-traversal
  const uint nTrvInvocs = 4u;
  const uint localTrvID = gl_SubgroupInvocationID % nTrvInvocs;
  const uint prevTrvID = uint(gl_SubgroupInvocationID + 1 - ((localTrvID == 3u) ? 4 : 0));
  const uint globalTrvID = gl_GlobalInvocationID.x / nTrvInvocs;
  
  // Wrapper structure for AtlasBuffer data
  struct AtlasLevel {
    uvec2 offset;
    uvec2 size;
  };

  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };
  
  // Wrapper structure for RootsBuffer data
  struct Roots {
    uint lidx;
    uint ridx;
    bool doSubdivide;
  };

  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 1, std430) restrict readonly buffer AtlasBuffer { AtlasLevel atlasLevels[]; };
  layout(binding = 2, std430) restrict readonly buffer RootsBuffer { Roots roots[]; };
  layout(binding = 0, r32f) coherent uniform image2DArray fieldImage;
  layout(location = 0) uniform uint nComps;
  layout(location = 1) uniform uint nArrayLevels;
  layout(location = 2) uniform uint topLevel;
  layout(location = 3) uniform float theta2;
  layout(location = 4) uniform sampler2D treeAtlasSampler;

  // Shared memory for texture atlas layout
  shared ivec2 shAtlasOffsets[nLevels];
  shared float shAtlasDiams2[nLevels];

  // Local memory for traversal
  uint queue[queueSize];
  ivec2 lloc;
  ivec2 rloc;
  uint lvl;
  bool rootsDiffer; // left tree root dn equal right tree root

  // Threads operate on a different image array level to reduce resource contention
  const uint arrayAdd = 3u * (gl_WorkGroupID.x % nArrayLevels); 

  /**
   * Atomic image addition to field images.
   * 
   * Splits provided components and atomically adds each to a separate bound image. This
   * is necessary as imageAtomicAdd(f32) does not come with multi-channel support.
   */
  void fieldAtomicAdd(in ivec2 loc, in vec3 v) {
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 0), v.x);
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 1), v.y);
    imageAtomicAdd(fieldImage, ivec3(loc, arrayAdd + 2), v.z);
  }

  /**
   * Read queue value at the current traversal level.
   */
  uint readQueue() {
    const uint shift = bool(lvl % 2) ? 16u : 0u;
    return (queue[lvl / 2] >> shift) & 0xffff;
  }

  /**
   * Modify queue value at the current traversal level, OR-ing it
   * with a value v.
   */
  void writeQueueOr(in uint v) {
    const uint shift = bool(lvl % 2) ? 16u : 0u;
    queue[lvl / 2] |= (v << shift);
  }

  /**
   * Modify queue value at the current traversal level, AND-ing it
   * with a value v.
   */
  void writeQueueAnd(in uint v) {
    const uint shift = bool(lvl % 2) ? 16u : 0u;
    queue[lvl / 2] &= ((v << shift) | (0xffff << (16u - shift)));
  }

  /**
   * Modify queue value at the current traversal level, OR-ing it
   * with a local cluster of 4 invocations in the subgroup.
   */
  void clusteredQueueOr() {
    queue[lvl / 2] = subgroupClusteredOr(queue[lvl / 2], 4u);
  }

  /**
   * Initialization step of dual-tree traversal.
   * 
   * Loads relevant data into shared memory, and acquires a pair
   * of root node indices between which we perform the actual
   * dual-tree traveral. Returns false if the current shader
   * invocation is out of bounds.
   */
  bool init() {
    // Read texture atlas layout into shared memory
    // Store squared diameter of tree nodes for BH approximation test
    const uint lid = gl_LocalInvocationIndex;
    if (lid < nLevels) {
      const AtlasLevel lvl = atlasLevels[lid];
      shAtlasOffsets[lid] = ivec2(lvl.offset);
      shAtlasDiams2[lid] = pow(length(vec2(bounds.range) / vec2(lvl.size)), 2);
    }
    barrier();

    // Get 1d indices of left and right root nodes we will compare
    // We precomputed these because of floating point inaccuracies in sqrt()
    const Roots roots = roots[globalTrvID];

    // Start traversal by descending tree root nodes
    const ivec2 dims = ivec2(atlasLevels[topLevel].size);
    lloc = ivec2(roots.lidx % dims.x, roots.lidx / dims.x) << 1;
    rloc = ivec2(roots.ridx % dims.x, roots.ridx / dims.x) << 1;
    lvl = topLevel - 1;
    rootsDiffer = lloc != rloc;
    
    // Valid traversal?
    return globalTrvID < nComps && roots.doSubdivide;
  }

  /**
   * Subdivision step of dual tree traversal.
   * 
   * Leverages shader subgroups to perform 4x4 node comparisons on a single level
   * of both trees. Four invocations each load in a left and right node, compare this, and
   * then rotate the right node among each other four times. If a node comparison succeeds,
   * forces are computed and added to the field. Otherwise, the comparison is put on the
   * queue for subdivision at a later time.
   */
  void subdivide() {
    // Left and right tree node locations
    const bool nodesDiffer = lloc != rloc || rootsDiffer;
    const float lvlDiams2 = shAtlasDiams2[lvl];
    const ivec2 _lloc = lloc + shAtlasOffsets[lvl] + ivec2(localTrvID % 2, localTrvID / 2);
    const ivec2 _rloc = rloc + shAtlasOffsets[lvl] + ivec2(localTrvID % 2, localTrvID / 2);

    // Read in node values
    const vec4 lnode = texelFetch(treeAtlasSampler, _lloc, 0);
    vec4 rnode = texelFetch(treeAtlasSampler, _rloc, 0);
    vec3 lwrite = vec3(0);
    vec3 rwrite = vec3(0);

    // Rotate the right node ri among four subgroup invocations
    uint ri = localTrvID;
    for (uint i = 0u; i < 4u; i++) {
      // Squared eucl. distance between nodes
      const vec2 t = lnode.yz - rnode.yz;
      const float t2 = dot(t, t);

      // If at least one node is empty, we truncate
      const bool isEmpty = lnode.x == 0.f || rnode.x == 0.f;

      // If node pair forms a suitable approximation, we truncate
      const bool isApprox = lvl == 0u
                          || (lnode.x == 1.f && rnode.x == 1.f)
                          || (lvlDiams2 / t2) < theta2;

      if (!isEmpty && isApprox && (nodesDiffer || localTrvID < ri)) {
        // Compute field values
        const float studt = 1.f / (1.f + t2);
        const vec2 studt2 = t * (studt * studt);

        // Add points x forces to both trees if they are different
        lwrite += rnode.x * vec3(studt, studt2);
        rwrite += lnode.x * vec3(studt, -studt2);
      } else if (!isEmpty && !isApprox && (nodesDiffer || localTrvID <= ri)) {
        // Comparison failed, push on queue for subdivision
        writeQueueOr(1u << (4u * localTrvID + ri));
      }
      
      // Perform rotate within four subgroup invocs
      rnode = subgroupShuffle(rnode, prevTrvID);
      rwrite = subgroupShuffle(rwrite, prevTrvID);
      ri = ++ri % 4u;
    }

    // Add node values to image
    if (lwrite != vec3(0)) {
      fieldAtomicAdd(_lloc, lwrite);
    }
    if (rwrite != vec3(0)) {
      fieldAtomicAdd(_rloc, rwrite);
    }

    // Combine the resulting four invocation queues together
    clusteredQueueOr();
  }

  /**
   * Traversal step of dual-tree traversal.
   * 
   * Checks if there are still comparisons left on the current level to 
   * subdivide. If there are none, it ascends until this is the case or 
   * we hit topLevel. If flagged comparison are found, we descend along 
   * the first such comparison and remove it from the queue. If no points 
   * are flagged whatsoever, returns false to stop traversal entirely.
   */
  bool traverse() {
    // Move upward to the first incomplete level
    while (readQueue() == 0u) {
      if (lvl == topLevel) {
        return false;
      }
      // Shift locs up one level
      lvl++;
      lloc >>= 1;
      rloc >>= 1;
    }
    
    // Pop next flagged comparison from queue
    const uint i = findMSB(readQueue());
    writeQueueAnd(~(1u << i));

    // li,ri order is 0,0 -> 0,1 -> 0,2 -> 0,3 -> 1,0 -> 1,1 -> 1,2 -> 1,3 -> ...
    const uint li = i / 4u;
    const uint ri = i % 4u;
    
    // Reset locs to leftmost position, then add offsets for this comparison
    // This only works for even locs, which is fine as the root nodes are 
    // handled separately.
    lloc = (lloc & ~1) + ivec2(li % 2u, li / 2u);
    rloc = (rloc & ~1) + ivec2(ri % 2u, ri / 2u);

    // Shift locs down one level
    lvl--;
    lloc <<= 1;
    rloc <<= 1;

    // Keep traversing
    return true;
  }
  
  void main() {
    // Obtain root nodes, and try to subdivide them
    if (!init()) {
      return;
    }
  
    // Traverse tree doing 4x4 node comparisons at each level
    do {
      subdivide();
    } while (traverse());
  }
);

/**
 * Dual-tree flattening shader.
 * 
 * Adds together the different results of the performed dual tree traversal, which
 * are located on different levels. Given that the number of levels can become
 * somewhat large, and each level requires three separate texel fetches, we opt
 * to perform a parallel reduction.
 * 
 * @author Mark van de Ruit
 */
GLSL(dt_flatten_reduce_src, 430,
  // Wrapper for AtlasBuffer data
  struct AtlasLevel {
    uvec2 offset;
    uvec2 size;
  };

  layout(local_size_x = 16, local_size_y = 16, local_size_z = 2) in;
  layout(std430, binding = 0) buffer AtlasBuffer { AtlasLevel atlasLevels[]; };
  layout(rgba32f, binding = 0) writeonly uniform image2D fieldImage;
  layout(location = 0) uniform uint nLevels;
  layout(location = 1) uniform uint nArrayLevels;
  layout(location = 2) uniform ivec2 fieldTextureSize;
  layout(location = 3) uniform sampler2DArray fieldAtlas;

  // Constants for shared memory and reduction
  const uint shSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
  const uint zGroupSize = gl_WorkGroupSize.z;
  const uint zHalfGroupSize = zGroupSize / 2;
  // Finest granularity is in the z-dimension which will be reduced by z invocations
  const uint shIdx = gl_LocalInvocationID.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z
                   + gl_LocalInvocationID.y * gl_WorkGroupSize.z 
                   + gl_LocalInvocationID.z;
  
  // Shared memory arrays
  shared ivec2 shAtlasOffsets[8]; // 8 is a guesstimate
  shared vec3 shForces[shSize];

  void main() {
    // Invocation location
    const uint lid = gl_LocalInvocationIndex;
    const uint zid = gl_LocalInvocationID.z;
    const ivec2 gxy = ivec2(gl_WorkGroupSize.xy 
                          * gl_WorkGroupID.xy
                          + gl_LocalInvocationID.xy);

    // Load atlas layout into shAtlasOffsets[] shared memory array
    if (lid < nLevels) {
      shAtlasOffsets[lid] = ivec2(atlasLevels[lid].offset);
    }
    barrier();

    // Invocations should only compute for valid pixels
    if (min(gxy, fieldTextureSize - 1) != gxy) {
      return;
    }

    // Initial sum, z active invocations
    vec3 v = vec3(0);
    for (uint j = zid; j < (nLevels * nArrayLevels); j += zGroupSize) {
      const uint lvl = j % nLevels;
      const uint arrayLvl = j / nLevels;
      const ivec2 xy = shAtlasOffsets[lvl] + (gxy >> lvl);
      v += vec3(texelFetch(fieldAtlas, ivec3(xy, 3u * arrayLvl + 0), 0).x,
                texelFetch(fieldAtlas, ivec3(xy, 3u * arrayLvl + 1), 0).x,
                texelFetch(fieldAtlas, ivec3(xy, 3u * arrayLvl + 2), 0).x);
    }
    shForces[shIdx] = v;
    barrier();
    if (zid < 1) {
      imageStore(fieldImage, gxy, vec4(shForces[shIdx] + shForces[shIdx + 1], 0));
    } 
  }
);