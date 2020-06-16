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

GLSL(field_src, 450,
  layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) buffer PosInterface { vec3 Positions[]; };
  layout(binding = 1, std430) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  layout(binding = 0, rgba32f) writeonly uniform image3D fields_texture;
  layout(location = 0) uniform uint num_points;
  layout(location = 1) uniform uint grid_depth;
  layout(location = 2) uniform uvec3 texture_size;
  layout(location = 3) uniform usampler2D grid_texture;

  // Reduction components
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared vec4 reductionArray[halfGroupSize];

  void main() {
    // Location of current workgroup
    ivec2 xyFixed = ivec2(gl_WorkGroupID.xy);
    vec2 xyUnfixed = (vec2(xyFixed) + vec2(0.5)) / vec2(texture_size.xy);
    uint lid = gl_LocalInvocationIndex.x;

    // Query grid map for entire z axis
    // Skip all pixels if empty
    uvec4 gridVec;
    if ((gridVec = texture(grid_texture, xyUnfixed)) == uvec4(0)) {
      uint z = gl_WorkGroupID.z * gl_WorkGroupSize.z + lid;
      if (z < texture_size.z) {
        imageStore(fields_texture, ivec3(xyFixed, z), vec4(0));
      }
      return;
    }
    
    // Workgroups stride over z-dimension
    for (uint z = gl_WorkGroupID.z; 
          z < texture_size.z; 
          z += gl_NumWorkGroups.z) {
      // Map xyz to [0, 1]
      ivec3 xyzFixed = ivec3(xyFixed, z);
      vec3 domain_pos = (vec3(xyzFixed) + vec3(0.5)) / vec3(texture_size);

      // Query grid map sample or skip pixel
      int _z = int(domain_pos.z * grid_depth);
      if (bitfieldExtract(gridVec[_z / 32], 31 - (_z % 32) , 1) == 0) {
        if (lid < 1) {
          imageStore(fields_texture, xyzFixed, vec4(0));
        }
        continue;
      }

      // Map to domain bounds
      domain_pos = domain_pos * range + minBounds;

      // Iterate over points to obtain density/gradient
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        vec3 t = domain_pos - Positions[i];
        float t_stud = 1.f / (1.f + dot(t, t));
        vec3 t_stud_2 = t * (t_stud * t_stud);

        // Field layout is: S, V.x, V.y, V.z
        v += vec4(t_stud, t_stud_2);
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
        vec4 reducedArray = reductionArray[0] + reductionArray[1];
        imageStore(fields_texture, xyzFixed, reducedArray);
      }
      barrier();
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

  layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
  
  layout(binding = 0, std430) restrict readonly buffer NodeBuffer { vec4 nodeBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer PosBuffer { vec3 posBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer DiamBuffer { vec4 diamBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;
  layout(binding = 0) uniform usampler2D gridSampler;

  layout(location = 0) uniform uint nPos;     // nr of points
  layout(location = 1) uniform uint kNode;    // Node fanout
  layout(location = 2) uniform uint nLvls;    // Nr of tree levels
  layout(location = 3) uniform uint gridDepth;
  layout(location = 4) uniform float theta;   // Approximation param
  layout(location = 5) uniform uvec3 textureSize;

  // Constants
  const uint logk = uint(log2(kNode));
  const uint bitmask = ~((~0u) << logk);
  const float theta2 = theta * theta;

  // Traversal local memory
  // We start a level lower, as the root node will never approximate
  uint lvl = 1u;
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

  bool approx(vec3 domainPos, inout vec4 fieldValue) {
    // Fetch packed node data
    const vec4 vNode = nodeBuffer[loc];
    const vec4 vDiam = diamBuffer[loc];

    // Unpack node data
    const vec3 center = vNode.xyz;
    const uint mass = uint(vNode.w);
    const vec3 diam = vDiam.xyz;
    const uint begin = uint(vDiam.w);

    // Squared distance to pos
    vec3 t = domainPos - center;
    float t2 = dot(t, t);

    // Compute squared diameter
    vec3 b = abs(normalize(t)) * vec3(-1, -1, 1);
    vec3 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);

      // Field layout is: S, V.x, V.y, V.z
      fieldValue += mass * vec4(tStud, t * (tStud * tStud));

      return true;
    } else if (mass <= 8 || lvl == nLvls - 1) {
      // Iterate over all leaf points (there goes thread divergence)
      for (uint i = begin; i < begin + mass; ++i) {
        t = domainPos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));

        // Field layout is: S, V.x, V.y, V.z
        fieldValue += vec4(tStud, t * (tStud * tStud));
       }
      return true;
    }
    return false;
  }

  vec4 traverse(vec3 domainPos) {
    vec4 fieldValue = vec4(0);

    do {
      if (approx(domainPos, fieldValue)) {
        ascend();
      } else {
        descend();
      }
    } while (lvl > 0u);

    return fieldValue;
  }

  void main() {
    // Invocation ID
    const uvec3 globalIdx = gl_WorkGroupID.xyz
                          * gl_WorkGroupSize.xyz
                          + gl_LocalInvocationID.xyz;
                   
    // Check that invocation is inside field texture
    if (min(globalIdx, textureSize - 1) != globalIdx) {
      return;
    }

    // Compute pixel position in [0, 1]
    vec3 domainPos = (vec3(globalIdx) + 0.5) / vec3(textureSize);

    // Read voxel grid to skip empty pixels
    uvec4 gridValue = texture(gridSampler, domainPos.xy);
    int z = int(domainPos.z * gridDepth);
    if (bitfieldExtract(gridValue[z / 32], 31 - (z % 32) , 1) == 0) {
      return;
    }

    // Map pixel position to domain bounds
    domainPos = domainPos * bounds.range + bounds.min;

    // Traverse tree and store result
    vec4 v = traverse(domainPos);
    imageStore(fieldImage, ivec3(globalIdx), v);
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