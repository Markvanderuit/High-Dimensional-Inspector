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

GLSL(stencil_vert_src, 450,
  layout(location = 0) in vec2 point;
  layout(std430, binding = 0) restrict readonly buffer Bounds { 
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };

  void main() {
    // Transform point into [0, 1] space
    vec2 position = (point - minBounds) * invRange;
    
    // Transform point into [-1, 1] clip space
    gl_Position =  vec4(position * 2.f - 1.f, 0.f, 1.f);
  }
);

GLSL(stencil_fragment_src, 450,
  layout(location = 0) out uint color;

  void main() {
    color = 1u;
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
  layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer Pos { vec2 Positions[]; };
  layout(binding = 1, std430) restrict readonly buffer Bounds { 
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fields_texture;
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uvec2 textureSize;
  layout(location = 2) uniform usampler2D stencilTexture;

  // Reduction components
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared vec3 reductionArray[halfGroupSize];

  void main() {
    // Location of current workgroup
    ivec2 xyFixed = ivec2(gl_WorkGroupID.xy);
    uint lid = gl_LocalInvocationIndex.x;

    // Skip pixel if stencil is empty
    if (texelFetch(stencilTexture, xyFixed, 0).x == 0u) {
      if (lid < 1) {
        imageStore(fields_texture, xyFixed, vec4(0));
      }
      return;
    } 

    // Map to domain pos
    vec2 domain_pos = (vec2(xyFixed) + vec2(0.5)) / vec2(textureSize);
    domain_pos = domain_pos * range + minBounds;

    // Iterate over points to obtain density/gradient
    vec3 v = vec3(0);
    for (uint i = lid; i < nPoints; i += groupSize) {
      vec2 t = domain_pos - Positions[i];
      float t_stud = 1.f / (1.f + dot(t, t));
      vec2 t_stud_2 = t * (t_stud * t_stud);

      // Field layout is: S, V.x, V.y
      v += vec3(t_stud, t_stud_2);
    }
    
    // Perform reduce add over all computed points for this pixel
    if (lid >= halfGroupSize) {
      reductionArray[lid - halfGroupSize] = v;
    }
    barrier();
    if (lid < halfGroupSize) {
      reductionArray[lid] += v;
    }
    for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
      barrier();
      if (lid < i) {
        reductionArray[lid] += reductionArray[lid + i];
      }
    }
    barrier();
    if (lid < 1) {
      vec3 reducedArray = reductionArray[0] + reductionArray[1];
      imageStore(fields_texture, xyFixed, vec4(reducedArray, 0));
    }
  }
);

GLSL(field_bvh_src, 450,
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Posit { vec2 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 4, std430) restrict readonly buffer Queue { uvec2 queueBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer QHead { uint queueHead; }; 
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  // Uniforms
  layout(location = 0) uniform uint nPos;         // nr of points
  layout(location = 1) uniform uint nLvls;        // Nr of tree levels
  layout(location = 2) uniform float theta2;      // Squared approximation param
  layout(location = 3) uniform uvec2 textureSize; // Size of fields texture

  // Fun stuff for fast modulo and division and stuff
  const uint bitmask = ~((~0u) << BVH_LOGK_2D);

  // Traversal data
  uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
  uint loc = 1u;
  uint stack = 1u | (bitmask << (BVH_LOGK_2D * lvl));

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

  bool approx(vec2 pos, inout vec3 field) {
    // Fetch node data for each invoc
    const vec4 node0 = node0Buffer[loc];
    const vec4 node1 = node1Buffer[loc];

    // Unpack node data
    const vec2 center = node0.xy;
    const uint mass = uint(node0.w);
    const vec2 diam = node1.xy;
    const uint begin = uint(node1.w);
    
    // Distance between node and pixel position
    const vec2 t = pos - center;
    const float t2 = dot(t, t);

    // Align the distance vector so it is always coming from the same part of the axis
    // such that the largest possible diameter of the bounding box is visible
    const vec2 b = abs(t) * vec2(-1, 1);
    const vec2 b_ = normalize(b);

    // Compute squared diameter, adjusted for viewing angle
    // This is the vector rejection of diam onto unit vector b_
    const vec2 ext = diam - b_ * dot(diam, b_); 

    // Tangens of the angle should not exceed theta
    if (dot(ext, ext) / t2 < theta2) { 
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);
      field += mass * vec3(tStud, t * (tStud * tStud));
      return true;
    } else if (mass <= EMB_BVH_KLEAF_2D || lvl == nLvls - 1) {
      // Iterate over all leaf points (hi, thread divergence here)
      for (uint i = begin; i < begin + mass; ++i) {
        vec2 t = pos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));
        field += vec3(tStud, t * (tStud * tStud));
      }
      return true;
    }
    return false;
  }

  vec3 traverse(vec2 pos) {
    vec3 field = vec3(0);

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
    const uvec2 px = queueBuffer[i];

    // Compute pixel position in [0, 1]
    // Then map pixel position to domain bounds
    vec2 pos = (vec2(px) + 0.5) / vec2(textureSize);
    pos = pos * bounds.range + bounds.min;

    // Traverse tree and store result
    imageStore(fieldImage, ivec2(px), vec4(traverse(pos), 0));
  }
);

GLSL(field_bvh_wide_src, 450, 
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require )

  // Wrapper structure for bounds data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Posit { vec2 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 4, std430) restrict readonly buffer Queue { uvec2 queueBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer QHead { uint queueHead; }; 
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  // Uniforms
  layout(location = 0) uniform uint nPos;         // nr of points
  layout(location = 1) uniform uint nLvls;        // Nr of tree levels
  layout(location = 2) uniform float theta2;      // Squared approximation param
  layout(location = 3) uniform uvec2 textureSize; // Size of fields texture

  // Fun stuff for fast modulo and division in our binary cases
  const uint logkQueue = uint(log2(32 >> BVH_LOGK_2D));
  const uint kNodeMod = bitfieldInsert(0, ~0, 0, int(BVH_LOGK_2D));
  const uint kQueueMod = bitfieldInsert(0, ~0, 0, int(logkQueue));

  // Traversal constants
  const uint thread = gl_LocalInvocationID.x & kNodeMod;
  const uint mask = bitfieldInsert(0, ~0, 0, int(BVH_KNODE_2D));

  // Traversal data
  uint queue[6]; // oops, hardcoded queue size :(
  uint lvl = 1u; // We start a level lower, as the root node will never approximate
  uint loc = 1u;

  // Read BVH_KNODE_2D bits from the current queue head
  uint readQueue() {
    const uint shift = (lvl & kQueueMod) << BVH_LOGK_2D;
    return mask & (queue[lvl >> logkQueue] >> shift);
  }

  // Bitwise OR BVH_KNODE_2D bits on the current queue head
  void orQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << BVH_LOGK_2D;
    queue[lvl >> logkQueue] |= v << shift;
  }

  // Bitwise AND BVH_KNODE_2D bits on the current queue head
  void andQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << BVH_LOGK_2D;
    queue[lvl >> logkQueue] &= (v << shift) | (~0 ^ (mask << shift)); // ouch
  }

  // Bitwise OR BVH_KNODE_2D bits on the current queue head between a cluster 
  // of BVH_KNODE_2D invocations 
  void clusterQueue() {
    const uint i = lvl >> logkQueue;
    queue[i] = subgroupClusteredOr(queue[i], BVH_KNODE_2D);
  }

  void computeField(in vec2 pos, inout vec3 field) {    
    // Fetch node data for each invoc
    const vec4 node0 = node0Buffer[loc + thread];
    const vec4 node1 = node1Buffer[loc + thread];

    // Unpack node data
    const vec2 center = node0.xy;
    const uint mass = uint(node0.w);
    const vec2 diam = node1.xy;
    const uint begin = uint(node1.w);

    // Compute squared distance between node and positio
    const vec2 t = pos - center;
    const float t2 = dot(t, t);

    // Align the distance vector so it is always coming from the same part of the axis
    // such that the largest possible diameter of the bounding box is visible
    const vec2 b = abs(t) * vec2(-1, 1);
    const vec2 b_ = normalize(b);

    // Compute squared diameter, adjusted for viewing angle
    // This is the vector rejection of diam onto unit vector b_
    const vec2 ext = diam - b_ * dot(diam, b_); 
    
    // Tangens of the angle should not exceed theta
    if (dot(ext, ext) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);
      field += mass * vec3(tStud, t * (tStud * tStud));
    } else if (mass <= EMB_BVH_KLEAF_2D || lvl == nLvls - 1) {
      // Iterate over all leaf points (hi, thread divergence here)
      for (uint i = begin; i < begin + mass; ++i) {
        vec2 t = pos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));
        field += vec3(tStud, t * (tStud * tStud));
      }
    } else {
      // Node must be subdivided, push flag on queue
      orQueue(1u << thread);
    }
   
    // Cluster queue values across subgroup
    clusterQueue();
  }

  vec3 traverse(vec2 pos) {
    // First tree level is always tested
    vec3 field = vec3(0);
    computeField(pos, field);

    // Traverse tree until root node is reached
    while (lvl != 0u) {
      const uint head = readQueue();
      if (head == 0u) {
        // Move up tree one level
        loc >>= BVH_LOGK_2D;
        lvl--;
      } else {
        // Pop next flagged node from queue head
        const uint flag = findLSB(head);
        andQueue(~(1u << flag));

        // Move down tree one level at flagged node
        loc = loc - ((loc - 1) & kNodeMod) + flag;
        loc = (loc * BVH_KNODE_2D) + 1u;
        lvl++;

        // Test full fan-out on new tree level
        computeField(pos, field);
      }
    }

    return field;
  }

  void main() {
    // Read pixel position from work queue. Make sure not to exceed queue head
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
                 / BVH_KNODE_2D;
    if (i >= queueHead) {
      return;
    }
    const uvec2 px = queueBuffer[i];

    // Compute pixel position in [0, 1]
    // Then map pixel position to domain bounds
    vec2 pos = (vec2(px) + 0.5) / vec2(textureSize);
    pos = pos * bounds.range + bounds.min;

    // Traverse tree, add together results from subgroup
    // Only let first thread in subgroup store result
    vec3 field = subgroupClusteredAdd(traverse(pos), BVH_KNODE_2D);
    if (thread == 0u) {
      imageStore(fieldImage, ivec2(i), vec4(field, 0));
    }
  }
);

GLSL(pixels_bvh_src, 450,
  // Return types for approx(...) function below
  #define STOP 0\n // Voxel encompasses node, stop traversal
  #define DESC 1\n // Node encompasses voxel, descend
  #define ASCN 2\n // Node and voxel dont overlap, ascend
  
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
  
  // Buffer, sampler and image bindings
  layout(binding = 0, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer MinbB { vec2 minbBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Bound { Bounds bounds; };
  layout(binding = 3, std430) restrict writeonly buffer Queue { uvec2 queueBuffer[]; };
  layout(binding = 4, std430) restrict coherent buffer QHead { uint queueHead; }; 

  // Uniforms
  layout(location = 0) uniform uint nPos;      // nr of points
  layout(location = 1) uniform uint nLvls;     // Nr of tree levels
  layout(location = 2) uniform uvec2 textureSize;

  // Fun stuff for fast modulo and division and stuff
  const uint bitmask = ~((~0u) << BVH_LOGK_2D);

  // Traversal data
  uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
  uint loc = 1u;
  uint stack = 1u | (bitmask << (BVH_LOGK_2D * lvl));

  // Pixel size
  const vec2 bbox = bounds.range / vec2(textureSize);
  const float extl = length(bbox);

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
    if (dist == 0) {
      loc++;
    } else {
      loc >>= BVH_LOGK_2D * dist;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = BVH_LOGK_2D * lvl;
    uint b = (stack >> shift) - 1;
    stack &= ~(bitmask << shift);
    stack |= b << shift;
  }

  bool overlap1D(vec2 a, vec2 b) {
    return a.y >= b.x && b.y >= a.x;
  }

  bool overlap2D(vec2 mina, vec2 maxa, vec2 minb, vec2 maxb) {
    return overlap1D(vec2(mina.x, maxa.x), vec2(minb.x, maxb.x))
        && overlap1D(vec2(mina.y, maxa.y), vec2(minb.y, maxb.y));
  }

  uint approx(vec2 minb, vec2 maxb, vec2 center) {
    // Fetch node data
    const vec2 _diam = node1Buffer[loc].xy;
    const vec2 _minb = minbBuffer[loc];
    const vec2 _maxb = _minb + _diam;
    const vec2 _center = _minb + 0.5 * _diam; 

    // Test bounding box containment
    const bool isOverlap = overlap2D(minb, maxb, _minb, _maxb);
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
    const uvec2 xy = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
    if (min(xy, textureSize - 1) != xy) {
      return;
    }

    // Compute information about this voxel
    vec2 center = (vec2(xy) + 0.5) / vec2(textureSize); // Map to [0, 1]
    center = bounds.min + bounds.range * center; // Map to embedding domain
    const vec2 minb = center - 1.5f * bbox; ;// - 0.5 * bbox;
    const vec2 maxb = center + 1.5f * bbox; ;// + 0.5 * bbox;

    // Traverse tree to find out if there is a closest or contained point
    do {
      const uint appr = approx(minb, maxb, center);
      if (appr == STOP) {
        // Store result and exit
        queueBuffer[atomicAdd(queueHead, 1)] = xy;
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
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer Pos { vec2 Positions[]; };
  layout(binding = 1, std430) restrict writeonly buffer Val { vec3 Values[]; };
  layout(binding = 2, std430) restrict readonly buffer Bounds { 
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };
  layout(location = 0) uniform uint num_points;
  layout(location = 1) uniform uvec2 texture_size;
  layout(location = 2) uniform sampler2D fields_texture;

  void main() {
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= num_points) {
      return;
    }

    // Map position of point to [0, 1]
    vec2 position = (Positions[i] - minBounds) * invRange;

    // Sample texture at mapped position
    Values[i] = texture(fields_texture, position).xyz;
  }
);