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

  out vec2 position;

  void main() {
    // Transform point into [0, 1] space
    position = (point.xy - minBounds) * invRange;

    // Transform point into [-1, 1] clip space
    gl_Position =  vec4(position * 2.f - 1.f, 0.f, 1.f);
  }
);

GLSL(density_lod_bottom_fragment, 430,
  layout (location = 0) out vec4 color;
  
  in vec2 position;

  void main() {
    // Blend together density and center of mass
    // by adding 1 for every point, and summing positions
    color = vec4(1.f, position, 0.f);
  }
);

GLSL(density_lod_bottom_divide_comp, 430,
  layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
  layout(std430, binding = 0) buffer AtlasDataBlock { 
    struct {
      uvec2 offset; 
      uvec2 size;
    } atlasLevelData[];
  };
  layout (rgba32f, binding = 0) writeonly uniform image2D level32f;

  uniform sampler2D texture32f;

  void main() {
    // Location of current workgroup unit
    ivec2 xy = ivec2(gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID);
    if (min(xy, ivec2(atlasLevelData[0].size) - 1) != xy) {
      return;
    }

    // Divide the center of mass by n
    vec4 v = texelFetch(texture32f, xy, 0);
    v.yz /= max(1.f, v.x); // Prevent divide-by-0 for empty spaces
    imageStore(level32f, xy, v);
  }
);

// Density computation for upper LOD levels, 2 steps
GLSL(density_lod_upper_src, 430,
  layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
  layout(std430, binding = 0) buffer AtlasDataBlock { 
    struct {
      uvec2 offset; 
      uvec2 size;
    } atlasLevelData[];
  };
  layout(rgba32f, binding = 0) writeonly uniform image2D level32f;
  
  uniform sampler2D texture32f;
  uniform int l;
  uniform int maxl;
  const ivec2 groupSize = ivec2(gl_WorkGroupSize.xy);
  const ivec2 halfGroupSize = groupSize / 2;
  shared vec4 reductionArray[64];

  void addTo(inout vec4 v, in vec4 add) {
    vec2 weights = vec2(v.x, add.x);
    float n = weights.x + weights.y;
    weights /= max(vec2(1), n); // Prevent divide-by-0

    vec2 xCenters = weights * vec2(v.y, add.y);
    vec2 yCenters = weights * vec2(v.z, add.z);

    float xv = xCenters.x + xCenters.y;
    float yv = yCenters.x + yCenters.y;
    
    v = vec4(n, xv, yv, 0);
  }

  vec4 sampleToplayer(ivec2 xy) {
    ivec2 prevxy = 2 * xy + ivec2(atlasLevelData[l - 1].offset);

    vec4 v = texelFetch(texture32f, prevxy, 0);
    addTo(v, texelFetch(texture32f, prevxy + ivec2(1, 0), 0));
    addTo(v, texelFetch(texture32f, prevxy + ivec2(0, 1), 0));
    addTo(v, texelFetch(texture32f, prevxy + ivec2(1, 1), 0));

    return v;
  }

  int toLid(ivec2 lxy) {
    return int(gl_WorkGroupSize.x) * lxy.y + lxy.x;
  }

  void main() {
    // Location of work unit
    const ivec2 xy = ivec2(gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID);
    const ivec2 lxy = ivec2(gl_LocalInvocationID.xy);
    const ivec2 lxyMod = lxy % 2;
    const uint lid = gl_LocalInvocationIndex;

    // Perform sample in 2x2 pixel area layer above and store result
    vec4 v = sampleToplayer(xy);
    {
      const ivec2 _xy = ivec2(atlasLevelData[l].offset) + xy;
      imageStore(level32f, _xy, v);
    }

    // Reduce add uneven to even if we still have levels left to compute
    if (l + 1 >= maxl) {
      return;
    }
    if (lxyMod.y != 0) {
      reductionArray[toLid(lxy - ivec2(0, 1))] = v;
    }
    barrier();
    if (lxyMod.y == 0) {
      addTo(reductionArray[lid], v);
    }
    barrier();
    if (lxyMod.x == 0) {
      addTo(reductionArray[lid], reductionArray[toLid(lxy + ivec2(1, 0))]);
    }
    barrier();
    if (lxyMod == ivec2(0)) {
      const ivec2 _xy = ivec2(atlasLevelData[l + 1].offset) + xy / 2;
      imageStore(level32f, _xy, reductionArray[lid]);
    }

    // Reduce add 2 to 0 if we still have levels left to compute
    if (l + 2 >= maxl) {
      return;
    }
    if (lxy.y == 0) {
      addTo(reductionArray[lid], reductionArray[toLid(lxy + ivec2(0, 2))]);
    }
    barrier();
    if (lxy == ivec2(0)) {
      addTo(reductionArray[0], reductionArray[toLid(lxy + ivec2(2, 0))]);
      const ivec2 _xy = ivec2(atlasLevelData[l + 2].offset) + xy / 4;
      imageStore(level32f, _xy, reductionArray[0]);
    }
  }
);