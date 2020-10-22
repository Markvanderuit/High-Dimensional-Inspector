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

GLSL(grid_vert_src, 450,
  layout(location = 0) in vec3 inVertex;
  layout(binding = 0, std430) restrict readonly buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  void main() {
    // Transform point into [0, 1] space
    vec3 position = (inVertex - minBounds) * invRange;
    
    // Transform point into [-1, 1] clip space
    gl_Position = vec4(position * 2.f - 1.f, 1.f);
  }
);

GLSL(grid_fragment_src, 450,
  layout(location = 0) out uvec4 outColor;
  layout(location = 0) uniform usampler1D cellMap;
  layout(location = 1) uniform float zPadding;
  layout(location = 2) uniform uint zDims;

  void main() {
    // Z value to look up cell values at
    const float zFixed = gl_FragCoord.z * zDims - 0.5f;
    
    // Fetch the two nearest cell values which cover this range
    const uvec4 lowerCell = texelFetch(cellMap, int(floor(zFixed)), 0);
    const uvec4 greaterCell = texelFetch(cellMap, int(ceil(zFixed)), 0);

    // Output logical or of these cells, flagging both their pixels
    outColor = lowerCell | greaterCell;
  }
);

GLSL(grid_flag_src, 450,
  layout(local_size_x = 8, local_size_y = 4, local_size_z = 4) in;
  
  layout(binding = 0, std430) restrict writeonly buffer Queue { uvec3 queueBuffer[]; };
  layout(binding = 1, std430) restrict coherent buffer QHead { uint queueHead; }; 

  layout(location = 0) uniform uint gridDepth;
  layout(location = 1) uniform uvec3 textureSize;
  layout(location = 2) uniform usampler2D gridSampler;
  
  void main() {
    // Check that invocation is inside field texture
    const uvec3 i = gl_WorkGroupID.xyz * gl_WorkGroupSize.xyz + gl_LocalInvocationID.xyz;
    if (min(i, textureSize - 1) != i) {
      return;
    }

    // Compute pixel position in [0, 1]
    vec3 pos = (vec3(i) + 0.5) / vec3(textureSize);

    // Read voxel grid
    uvec4 gridValue = texture(gridSampler, pos.xy);
    int z = int(pos.z * gridDepth);

    // Push pixel on queue if not to be skipped
    if (bitfieldExtract(gridValue[z / 32], 31 - (z % 32) , 1) > 0) {
      queueBuffer[atomicAdd(queueHead, 1)] = i;
    }
  }
);

GLSL(divide_dispatch_src, 450,
  layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer Value { uint value; };
  layout(binding = 1, std430) restrict writeonly buffer Disp { uvec3 dispatch; };
  layout(location = 0) uniform uint div;

  void main() {
    dispatch = uvec3((value + div - 1) / div, 1, 1);
  }
);

GLSL(field_src, 450,
  // Wrapper structure for Bounds data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer object bindings
  layout(binding = 0, std430) restrict readonly buffer Posit { vec3 posBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 2, std430) restrict readonly buffer Queue { uvec3 queueBuffer[]; };
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

  // Uniform locations
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uvec3 textureSize;

  // Shared memory for reduction
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared vec4 reductionArray[halfGroupSize];
  
  // Shared memory for pixel position
  shared uvec3 px;

  void main() {
    // Read pixel position from work queue. Make sure not to exceed queue head
    const uint i = gl_WorkGroupID.x;
    const uint li = gl_LocalInvocationID.x;
    if (li == 0) {
      px = queueBuffer[i];
    }
    barrier();

    // Compute pixel position in [0, 1]
    // Then map pixel position to domain bounds
    vec3 pos = (vec3(px) + 0.5) / vec3(textureSize);
    pos = pos * bounds.range + bounds.min;

    // Iterate over points to obtain density/gradient field values
    vec4 field = vec4(0);
    for (uint j = li; j < nPoints; j += groupSize) {
      // Field layout is: S, V.x, V.y, V.z
      vec3 t = pos - posBuffer[j];
      float tStud = 1.f / (1.f + dot(t, t));
      field += vec4(tStud, t * (tStud * tStud));
    }
    
    // Perform reduce add over all computed points for this pixel
    if (li >= halfGroupSize) {
      reductionArray[li - halfGroupSize] = field;
    }
    barrier();
    if (li < halfGroupSize) {
      reductionArray[li] += field;
    }
    for (uint j = halfGroupSize / 2; j > 1; j /= 2) {
      barrier();
      if (li < j) {
        reductionArray[li] += reductionArray[li + j];
      }
    }
    barrier();

    if (li == 0) {
      vec4 reducedArray = reductionArray[0] + reductionArray[1];
      imageStore(fieldImage, ivec3(px), reducedArray);
    }
  }
);

GLSL(field_bvh_src, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Posit { vec3 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 4, std430) restrict readonly buffer Queue { uvec3 queueBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer QHead { uint queueHead; };
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

  // Uniforms
  layout(location = 0) uniform uint nPos;      // nr of points
  layout(location = 1) uniform uint nLvls;     // Nr of tree levels
  layout(location = 2) uniform float theta2;   // Squared approximation param
  layout(location = 3) uniform uvec3 textureSize;

  // Fun stuff for fast modulo and division and stuff
  const uint bitmask = ~((~0u) << BVH_3D_LOGK);

  // Traversal data
  uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
  uint loc = 1u;
  uint stack = 1u | (bitmask << (BVH_3D_LOGK * lvl));

  void descend() {
    // Move down tree
    lvl++;
    loc = loc * BVH_3D_KNODE + 1u;

    // Push unvisited locations on stack
    stack |= (bitmask << (BVH_3D_LOGK * lvl));
  }

  void ascend() {
    // Find distance to next level on stack
    uint nextLvl = findMSB(stack) / BVH_3D_LOGK;
    uint dist = lvl - nextLvl;

    // Move dist up to where next available stack position is
    // and one to the right
    if (dist == 0) {
      loc++;
    } else {
      loc >>= BVH_3D_LOGK * dist;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = BVH_3D_LOGK * lvl;
    uint b = (stack >> shift) - 1;
    stack &= ~(bitmask << shift);
    stack |= b << shift;
  }

  bool approx(vec3 pos, inout vec4 field) {
    // Fetch node data for each invoc
    const vec4 node0 = node0Buffer[loc];
    const vec4 node1 = node1Buffer[loc];

    // Unpack node data
    const vec3 center = node0.xyz;
    const uint mass = uint(node0.w);
    const vec3 diam = node1.xyz;
    const uint begin = uint(node1.w);

    // Squared distance to pos
    vec3 t = pos - center;
    float t2 = dot(t, t);

    // Align the distance vector so it is always coming from the same part of the axis
    // such that the largest possible diameter of the bounding box is visible
    const vec3 b = abs(t) * vec3(-1, -1, 1);
    const vec3 b_ = normalize(b);

    // Compute squared diameter, adjusted for viewing angle
    // This is the vector rejection of diam onto unit vector b_
    const vec3 ext = diam - b_ * dot(diam, b_); 

    // if (pTheta * pTheta > theta2) {
    if (dot(ext, ext) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);
      field += mass * vec4(tStud, t * (tStud * tStud));
      return true;
    } else if (mass <= EMB_BVH_3D_KLEAF || lvl == nLvls - 1) {
      // Leaf is reached. iterate over all data. The dual
      // hierarchy handles this much better, as it can
      // just deal with large leaves in a separate shader...
      for (uint i = begin; i < begin + mass; ++i) {
        vec3 t = pos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));
        field += vec4(tStud, t * (tStud * tStud));
      }
      return true;
    }
    return false;
  }

  vec4 traverse(vec3 pos) {
    vec4 field = vec4(0);

    do {
      if (approx(pos, field)) {
        ascend();
      } else {
        descend();
      }
    } while (lvl > 0u);

    return field;
  }

  void main() {
    // Read pixel position from work queue. Make sure not to exceed queue head
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= queueHead) {
      return;
    }
    const uvec3 px = queueBuffer[i];

    // Compute pixel position in [0, 1]
    // Then map pixel position to domain bounds
    vec3 pos = (vec3(px) + 0.5) / vec3(textureSize);
    pos = pos * bounds.range + bounds.min;

    // Traverse tree and store result
    imageStore(fieldImage, ivec3(px), traverse(pos));
  }
);

GLSL(field_bvh_wide_src, 450,
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support

  // Wrapper structure for bounds data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Posit { vec3 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 4, std430) restrict readonly buffer Queue { uvec3 queueBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer QHead { uint queueHead; }; 
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

  // Uniforms
  layout(location = 0) uniform uint nPos;      // nr of points
  layout(location = 1) uniform uint nLvls;     // Nr of tree levels
  layout(location = 2) uniform float theta2;   // Squared approximation param
  layout(location = 3) uniform uvec3 textureSize;

  // Fun stuff for fast modulo and division in our binary cases
  const uint logkQueue = uint(log2(32 >> BVH_3D_LOGK));
  const uint kNodeMod = bitfieldInsert(0, ~0, 0, int(BVH_3D_LOGK));
  const uint kQueueMod = bitfieldInsert(0, ~0, 0, int(logkQueue));

  // Traversal constants
  const uint thread = gl_LocalInvocationID.x & kNodeMod;
  const uint mask = bitfieldInsert(0, ~0, 0, int(BVH_3D_KNODE));

  // Traversal data
  uint queue[6]; // oops, hardcoded queue size :(
  uint lvl = 1u; // We start a level lower, as the root node will never approximate
  uint loc = 1u;

  // Read BVH_3D_KNODE bits from the current queue head
  uint readQueue() {
    const uint shift = (lvl & kQueueMod) << BVH_3D_LOGK;
    return mask & (queue[lvl >> logkQueue] >> shift);
  }

  // Bitwise OR BVH_3D_KNODE bits on the current queue head
  void orQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << BVH_3D_LOGK;
    queue[lvl >> logkQueue] |= v << shift;
  }

  // Bitwise AND BVH_3D_KNODE bits on the current queue head
  void andQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << BVH_3D_LOGK;
    queue[lvl >> logkQueue] &= (v << shift) | (~0 ^ (mask << shift)); // ouch
  }

  // Bitwise OR BVH_3D_KNODE bits on the current queue head between a cluster 
  // of BVH_3D_KNODE invocations 
  void clusterQueue() {
    const uint i = lvl >> logkQueue;
    queue[i] = subgroupClusteredOr(queue[i], BVH_3D_KNODE);
  }

  void computeField(in vec3 pos, inout vec4 field) {
    // Fetch node data for each invoc
    const vec4 node0 = node0Buffer[loc + thread];
    const vec4 node1 = node1Buffer[loc + thread];

    // Unpack node data
    const vec3 center = node0.xyz;
    const uint mass = uint(node0.w);
    const vec3 diam = node1.xyz;
    const uint begin = uint(node1.w);

    // Compute squared distance between node and position
    vec3 t = pos - center;
    float t2 = dot(t, t);

    // Compute squared diameter, adjusted for viewing angle
    const vec3 b = abs(normalize(t)) * vec3(-1, -1, 1);
    const vec3 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);
      field += mass * vec4(tStud, t * (tStud * tStud));
    } else if (mass <= EMB_BVH_3D_KLEAF || lvl == nLvls - 1) {
      // Iterate over all leaf points (there goes thread divergence)
      for (uint i = begin; i < begin + mass; ++i) {
        vec3 t = pos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));
        field += vec4(tStud, t * (tStud * tStud));
      }
    } else {
      // Node must be subdivided, push flag on queue
      orQueue(1u << thread);
    }
   
    // Cluster queue values across subgroup
    clusterQueue();
  }

  vec4 traverse(vec3 pos) {
    // First tree level is always tested
    vec4 field = vec4(0);
    computeField(pos, field);

    // Traverse tree until root node is reached
    while (lvl != 0u) {
      const uint head = readQueue();
      if (head == 0u) {
        // Move up tree one level
        loc = (loc - 1) >> BVH_3D_LOGK;
        // loc >>= BVH_3D_LOGK;
        lvl--;
      } else {
        // Pop next flagged node from queue head
        const uint flag = findLSB(head);
        andQueue(~(1u << flag));

        // Move down tree one level at flagged node
        loc = loc - ((loc - 1) & kNodeMod) + flag;
        loc = (loc * BVH_3D_KNODE) + 1u;
        lvl++;

        // Test full fan-out on new tree level
        computeField(pos, field);
      }
    }

    return field;
  }

  void main() {
    // Read pixel position from work queue. Make sure not to exceed queue head
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) 
                 / BVH_3D_KNODE; // Each pixel is worked on by BVH_3D_KNODE invocations
    if (i >= queueHead) {
      return;
    }
    const uvec3 px = queueBuffer[i];

    // Compute pixel position in [0, 1]
    // Then map pixel position to domain bounds
    vec3 pos = (vec3(px) + 0.5) / vec3(textureSize);
    pos = pos * bounds.range + bounds.min;

    // Traverse tree, add together results from subgroup
    // Only let first thread in subgroup store result
    vec4 field = subgroupClusteredAdd(traverse(pos), BVH_3D_KNODE);
    if (thread == 0u) {
      imageStore(fieldImage, ivec3(px), field);
    }
  }
);

GLSL(flag_bvh_src, 450,
  // Return types for approx(...) function below
  #define STOP 0\n // Voxel encompasses node, stop traversal
  #define DESC 1\n // Node encompasses voxel, descend
  #define ASCN 2\n // Node and voxel dont overlap, ascend
  
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in;
  
  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer MinbB { vec3 minbBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 3, std430) restrict writeonly buffer Queue { uvec3 queueBuffer[]; };
  layout(binding = 4, std430) restrict coherent buffer QHead { uint queueHead; }; 

  // Uniforms
  layout(location = 0) uniform uint nPos;      // nr of points
  layout(location = 1) uniform uint nLvls;     // Nr of tree levels
  layout(location = 2) uniform uvec3 textureSize;

  // Fun stuff for fast modulo and division and stuff
  const uint bitmask = ~((~0u) << BVH_3D_LOGK);

  // Traversal data
  uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
  uint loc = 1u;
  uint stack = 1u | (bitmask << (BVH_3D_LOGK * lvl));

  // Pixel size
  const vec3 bbox = bounds.range / vec3(textureSize);
  const float extl = length(bbox);

  void descend() {
    // Move down tree
    lvl++;
    loc = loc * BVH_3D_KNODE + 1u;

    // Push unvisited locations on stack
    stack |= (bitmask << (BVH_3D_LOGK * lvl));
  }

  void ascend() {
    // Find distance to next level on stack
    uint nextLvl = findMSB(stack) / BVH_3D_LOGK;
    uint dist = lvl - nextLvl;

    // Move dist up to where next available stack position is
    // and one to the right
    if (dist == 0) {
      loc++;
    } else {
      loc >>= BVH_3D_LOGK * dist;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = BVH_3D_LOGK * lvl;
    uint b = (stack >> shift) - 1;
    stack &= ~(bitmask << shift);
    stack |= b << shift;
  }

  bool overlap1D(vec2 a, vec2 b) {
    return a.y >= b.x && b.y >= a.x;
  }

  bool overlap3D(vec3 mina, vec3 maxa, vec3 minb, vec3 maxb) {
    return overlap1D(vec2(mina.x, maxa.x), vec2(minb.x, maxb.x))
        && overlap1D(vec2(mina.y, maxa.y), vec2(minb.y, maxb.y))
        && overlap1D(vec2(mina.z, maxa.z), vec2(minb.z, maxb.z));
  }

  uint approx(vec3 minb, vec3 maxb, vec3 center) {
    // Fetch node data
    const vec3 _diam = node1Buffer[loc].xyz;
    const vec3 _minb = minbBuffer[loc];
    const vec3 _maxb = _minb + _diam;
    const vec3 _center = _minb + 0.5 * _diam; 

    // Test bounding box containment
    const bool isOverlap = overlap3D(minb, maxb, _minb, _maxb);
    const bool isMin = min(_minb, minb) == minb;
    const bool isMax = max(_maxb, maxb) == maxb;

    if (isOverlap) {
      if (lvl == nLvls - 1 || isMin && isMax) {
        return STOP;
      } else {
        return DESC;
      }
    } else {
      return ASCN;
    }
  }

  void main() {
    const uvec3 xyz = gl_WorkGroupID.xyz * gl_WorkGroupSize.xyz + gl_LocalInvocationID.xyz;
    if (min(xyz, textureSize - 1) != xyz) {
      return;
    }

    // Compute information about this voxel
    vec3 center = ((vec3(xyz) + 0.5) / vec3(textureSize)); // Map to [0, 1]
    center = bounds.min + bounds.range * center; // Map to embedding domain
    const vec3 minb = center - 1.5f * bbox; ;// - 0.5 * bbox;
    const vec3 maxb = center + 1.5f * bbox; ;// + 0.5 * bbox;

    // Traverse tree to find out if there is a closest or contained point
    do {
      const uint appr = approx(minb, maxb, center);
      if (appr == STOP) {
        // Store result and exit
        queueBuffer[atomicAdd(queueHead, 1)] = xyz;
        return;
      } else if (appr == ASCN) {
        ascend();
      } else if (appr == DESC) {
        descend();
      } else {
        return;
      }
    } while (lvl > 0);
  }
);

GLSL(interp_src, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) buffer PosBuffer { vec3 positions[]; };
  layout(binding = 1, std430) buffer ValueBuffer { vec4 values[]; };
  layout(binding = 2, std430) buffer BoundsBuffer { Bounds bounds; };
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uvec3 fieldSize;
  layout(location = 2) uniform sampler3D fieldSampler;

  shared vec3 minBounds;
  shared vec3 invRange;

  void main() {
    // Invocation ID
    const uint lid = gl_LocalInvocationID.x;
    const uint gid = gl_WorkGroupID.x
                   * gl_WorkGroupSize.x
                   + gl_LocalInvocationID.x;

    // Load shared memory
    if (lid < 1) {
      minBounds = bounds.min;
      invRange = bounds.invRange;
    }
    barrier();

    // Only invocations under nPoints should proceed
    if (gid >= nPoints) {
      return;
    }

    // Map position of point to [0, 1]
    vec3 pos = (positions[gid] - minBounds) * invRange;

    // Sample texture at mapped position
    values[gid] = texture(fieldSampler, pos);
  }
);