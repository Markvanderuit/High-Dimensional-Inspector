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

GLSL(density_lod_bottom_vertex, 430,
  layout(location = 0) in vec2 point;
  layout(std430, binding = 0) buffer BoundsInterface { 
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };

  layout(location = 0) out vec2 position;

  void main() {
    // Transform point into [0, 1] space
    position = (point.xy - minBounds) * invRange;

    // Transform point into [-1, 1] clip space
    gl_Position =  vec4(position * 2.f - 1.f, 0.f, 1.f);
  }
);

GLSL(density_lod_bottom_fragment, 430,
  layout(location = 0) in vec2 position;
  layout(location = 0) out vec4 color;

  void main() {
    // Blend together density and center of mass
    // by adding 1 for every point, and summing positions
    color = vec4(1.f, position, 0.f);
  }
);

GLSL(density_lod_bottom_divide_comp, 430,
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

  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer AtlasBuffer { AtlasLevel atlasLevels[]; };
  layout(binding = 1, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 0, rgba32f) restrict uniform image2D level32f;

  shared ivec2 shAtlasLevelSize;

  void main() {
    // Location of current invocation
    const ivec2 gxy = ivec2(gl_WorkGroupID 
                          * gl_WorkGroupSize 
                          + gl_LocalInvocationID);
    const uint lid = gl_LocalInvocationIndex;

    // Load shared memory
    if (lid == 0) {
      shAtlasLevelSize = ivec2(atlasLevels[0].size);
    }
    barrier();

    // Only proceed with invocations inside image bounds
    if (min(gxy, shAtlasLevelSize - 1) != gxy) {
      return;
    }

    // Load image value at current position
    vec4 v = imageLoad(level32f, gxy);

    // Modify image value:
    // * Divide the center of mass by n,
    // * Transform center of mass into embedding domain
    v.yz /= max(1.f, v.x); // Prevent divide-by-0 for empty space
    v.yz = v.yz * bounds.range + bounds.min;

    // Write image value back at current position
    imageStore(level32f, gxy, v);
  }
);

// Density computation for upper LOD levels, 2 steps
GLSL(density_lod_upper_src, 430,
  // Wrapper structure for AtlasBuffer data
  struct AtlasLevel {
    uvec2 offset;
    uvec2 size;
  };

  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer AtlasDataBlock { AtlasLevel atlasLevels[]; };
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D level32f;
  layout(location = 0) uniform sampler2D texture32f;
  layout(location = 1) uniform uint l;
  layout(location = 2) uniform uint maxl;
  layout(location = 3) uniform uint nLevels;
  
  shared vec4 reductionArray[gl_WorkGroupSize.x * gl_WorkGroupSize.y];
  shared AtlasLevel shAtlasLevels[12];

  vec4 merge(in vec4 v1, in vec4 v2) {
    // New number of points
    float n = v1.x + v2.x; // Combined number of points

    // New center of mass
    vec2 weights = vec2(v1.x, v2.x);
    weights /= max(vec2(1), n); // Prevent divide-by-0 for empty space
    vec2 c = weights.x * v1.yz + weights.y * v2.yz;

    return vec4(n, c, 0);
  }

  bool isBelow(ivec2 xy, ivec2 maxBounds) {
    return min(xy, maxBounds) == xy;
  }

  vec4 sampleToplayer(ivec2 gxy) {
    const ivec2 offset = ivec2(shAtlasLevels[l - 1].offset);
    const ivec2 xy = offset + 2 * gxy;

    // Sample four top-level texels
    const vec4 v0 = texelFetch(texture32f, xy, 0);
    const vec4 v1 = texelFetch(texture32f, xy + ivec2(1, 0), 0);
    const vec4 v2 = texelFetch(texture32f, xy + ivec2(0, 1), 0);
    const vec4 v3 = texelFetch(texture32f, xy + 1, 0);

    // Normalize weights
    vec4 weights = vec4(v0.x, v1.x, v2.x, v3.x);
    float n = dot(weights, vec4(1));
    weights /= max(n, 1.f);

    // Compute center of mass
    vec2 center = vec2(dot(weights, vec4(v0.y, v1.y, v2.y, v3.y)), 
                       dot(weights, vec4(v0.z, v1.z, v2.z, v3.z)));

    return vec4(n, center, 0);
  }

  void main() {
    // Location of work unit
    const ivec2 xy = ivec2(gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID);
    const ivec2 lxy = ivec2(gl_LocalInvocationID.xy);
    const uint lid = gl_LocalInvocationIndex;

    if (lid < nLevels) {
      shAtlasLevels[lid] = atlasLevels[lid];
    }
    barrier();

    // Perform sample in 2x2 pixel area in l - 1, and store result in l
    vec4 v = sampleToplayer(xy);
    if (isBelow(xy, ivec2(shAtlasLevels[l].size) - 1)) {
      const ivec2 _xy = ivec2(shAtlasLevels[l].offset) + xy;
      imageStore(level32f, _xy, v);
    }
    reductionArray[lid] = v;
    barrier();

    // Reduce add using modulo, dropping 3/4th of remaining pixels every l + 1
    int modOffset = 1;
    int modFactor = 2;
    for (uint _l = l + 1; _l < maxl; _l++) {
      const ivec2 lxyMod = lxy % modFactor;
      const ivec2 xyh = lxy + ivec2(modOffset, 0);
      const ivec2 xyv = lxy + ivec2(0, modOffset);
      const uint lidh = gl_WorkGroupSize.x * xyh.y + xyh.x;
      const uint lidv = gl_WorkGroupSize.x * xyv.y + xyv.x;

      // Horizontal shift
      if (lxyMod.x == 0) {
        reductionArray[lid] = merge(reductionArray[lid], reductionArray[lidh]);
      }
      barrier();

      // Vertical shift
      if (lxyMod.y == 0) {
        reductionArray[lid] = merge(reductionArray[lid], reductionArray[lidv]);
      }
      barrier();

      // Store result in atlas layer _l
      ivec2 _xy = xy / modFactor;
      if (lxyMod == ivec2(0) && isBelow(_xy, ivec2(shAtlasLevels[_l].size) - 1)) {
        _xy += ivec2(shAtlasLevels[_l].offset);
        imageStore(level32f, _xy, reductionArray[lid]);
      }

      modOffset *= 2;
      modFactor *= 2;
    }
  }
);