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
#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

GLSL(grid_vert_src, 450,
  layout(location = 0) in vec4 point;
  layout(std430, binding = 0) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  void main() {
    // Transform point into [0, 1] space
    vec3 position = (point.xyz - minBounds) * invRange;
    
    // Transform point into [-1, 1] clip space
    gl_Position =  vec4(position * 2.f - 1.f, 1.f);
  }
);

GLSL(grid_fragment_src, 450,
  layout(location = 0) out uvec4 color;
  layout(location = 0) uniform usampler1D cellMap;
  layout(location = 1) uniform float zPadding;

  void main() {
    // Z value to look up cell values at
    float zFixed = gl_FragCoord.z * textureSize(cellMap, 0).x - 0.5f;
    
    // Store OR of the two nearest cells as result
    uvec4 lowerCell = texelFetch(cellMap, int(floor(zFixed)), 0);
    uvec4 greaterCell = texelFetch(cellMap, int(ceil(zFixed)), 0);
    color = lowerCell | greaterCell;
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
  layout(location = 2) uniform uint kNode;     // Node fanout
  layout(location = 3) uniform uint kLeaf;     // Leaf fanout
  layout(location = 4) uniform float theta2;   // Squared approximation param
  layout(location = 5) uniform uvec3 textureSize;

  // Fun stuff for fast modulo and division and stuff
  const uint logk = uint(log2(kNode));
  const uint bitmask = ~((~0u) << logk);

  // Traversal data
  uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
  uint loc = 1u;
  uint stack = 1u | (bitmask << (logk * lvl));

  void descend() {
    // Move down tree
    lvl++;
    loc = loc * kNode + 1u;

    // Push unvisited locations on stack
    stack |= (bitmask << (logk * lvl));
  }

  void ascend() {
    // Find distance to next level on stack
    uint nextLvl = findMSB(stack) / logk;
    uint dist = lvl - nextLvl;

    // Move dist up to where next available stack position is
    // and one to the right
    if (dist == 0) {
      loc++;
    } else {
      loc >>= logk * dist;
    }
    lvl = nextLvl;

    // Pop visited location from stack
    uint shift = logk * lvl;
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

    // Compute squared diameter
    vec3 b = abs(normalize(t)) * vec3(-1, -1, 1);
    vec3 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);
      field += mass * vec4(tStud, t * (tStud * tStud));
      return true;
    } else if (mass <= kLeaf || lvl == nLvls - 1) {
      // Iterate over all leaf points (there goes thread divergence)
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
  #extension GL_KHR_shader_subgroup_clustered : require\n

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
  layout(location = 2) uniform uint kNode;     // Node fanout
  layout(location = 3) uniform uint kLeaf;     // Leaf fanout
  layout(location = 5) uniform float theta2;   // Squared approximation param
  layout(location = 6) uniform uvec3 textureSize;

  // Fun stuff for fast modulo and division in our binary cases
  const uint logk = uint(log2(kNode));
  const uint logKQueue = uint(log2(32 >> logk));
  const uint kNodeMod = bitfieldInsert(0, ~0, 0, int(logk));
  const uint kQueueMod = bitfieldInsert(0, ~0, 0, int(logKQueue));

  // Traversal constants
  const uint thread = gl_LocalInvocationID.x & kNodeMod;
  const uint mask = bitfieldInsert(0, ~0, 0, int(kNode));

  // Traversal data
  uint queue[6]; // oops, hardcoded queue size :(
  uint lvl = 1u; // We start a level lower, as the root node will never approximate
  uint loc = 1u;

  // Read kNode bits from the current queue head
  uint readQueue() {
    const uint shift = (lvl & kQueueMod) << logk;
    return mask & (queue[lvl >> logKQueue] >> shift);
  }

  // Bitwise OR kNode bits on the current queue head
  void orQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << logk;
    queue[lvl >> logKQueue] |= v << shift;
  }

  // Bitwise AND kNode bits on the current queue head
  void andQueue(in uint v) {
    const uint shift = (lvl & kQueueMod) << logk;
    queue[lvl >> logKQueue] &= (v << shift) | (~0 ^ (mask << shift)); // ouch
  }

  // Bitwise OR kNode bits on the current queue head between a cluster 
  // of kNode invocations 
  void clusterQueue() {
    const uint i = lvl >> logKQueue;
    queue[i] = subgroupClusteredOr(queue[i], kNode);
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
    } else if (mass <= kLeaf || lvl == nLvls - 1) {
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
        loc >>= logk;
        lvl--;
      } else {
        // Pop next flagged node from queue head
        const uint flag = findLSB(head);
        andQueue(~(1u << flag));

        // Move down tree one level at flagged node
        loc = loc - ((loc - 1) & kNodeMod) + flag;
        loc = (loc * kNode) + 1u;
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
                 / kNode; // Each pixel is worked on by kNode invocations
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
    vec4 field = subgroupClusteredAdd(traverse(pos), kNode);
    if (thread == 0u) {
      imageStore(fieldImage, ivec3(px), field);
    }
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