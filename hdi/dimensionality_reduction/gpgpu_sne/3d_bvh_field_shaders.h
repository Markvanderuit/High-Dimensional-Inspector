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

// Vertex shader for voxel grid computation
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

// Fragment shader for voxel grid computation
GLSL(grid_fragment_src, 450,
  layout (location = 0) out uvec4 color;

  uniform usampler1D cellMap;
  uniform float zPadding;

  void main() {
    // Z value to look up cell values at
    float zFixed = gl_FragCoord.z * textureSize(cellMap, 0).x - 0.5f;
    
    // Store OR of the two nearest cells as result
    uvec4 lowerCell = texelFetch(cellMap, int(floor(zFixed)), 0);
    uvec4 greaterCell = texelFetch(cellMap, int(ceil(zFixed)), 0);
    color = lowerCell | greaterCell;
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
  layout(binding = 1, std430) restrict readonly buffer IdxBuffer { uint idxBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer PosBuffer { vec4 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer MinbBuffer { vec3 minbBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer DiamBuffer { vec3 diamBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
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
    // Query values from current node
    vec4 node = nodeBuffer[loc];
    vec3 diam = diamBuffer[loc];
    
    // Squared distance to pos
    vec3 t = domainPos - node.xyz;
    float t2 = dot(t, t);

    // Compute squared diameter
    vec3 b = abs(normalize(t)) * vec3(-1, -1, 1);
    vec3 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2 || node.w < 2.f) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);

      // Field layout is: S, V.x, V.y, V.z
      fieldValue += node.w * vec4(tStud, t * (tStud * tStud));

      return true;
    } else if (lvl == nLvls - 1) {
      // If a leaf node is reached that is not approximate enough
      // we iterate over all contained points (there goes thread divergence)
      uint begin = idxBuffer[loc];
      for (uint i = begin; i < begin + uint(node.w); ++i) {
        t = domainPos - posBuffer[i].xyz;
        float tStud = 1.f / (1.f +  dot(t, t));

        // Field layout is: S, V.x, V.y, V.z
        fieldValue += vec4(tStud, t * (tStud * tStud));
       }

      return true;
    }
    
    return false;

    // // Truncate if... 
    // bool trunc = (mass <= 1.f)            // Empty or leaf node
    //           || lvl >= nLvls - 1         // Leaf node
    //           || (node.w / t2 < theta2);  // Approximation
    
    // // Compute forces
    // if (trunc && mass > 0.f) {
    //   float tStud = 1.f / (1.f + t2);
    //   vec3 tStud2 = t * (tStud * tStud);

    //   // Field layout is: S, V.x, V.y, V.z
    //   fieldValue += mass * vec4(tStud, tStud2);
    // }

    // return trunc;
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

// Compute shader for point sampling from field
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