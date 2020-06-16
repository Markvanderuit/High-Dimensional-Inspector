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
  layout(location = 0) uniform uint num_points;
  layout(location = 1) uniform uvec2 texture_size;
  layout(location = 2) uniform usampler2D stencil_texture;

  // Reduction components
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared vec3 reductionArray[halfGroupSize];

  void main() {
    // Location of current workgroup
    ivec2 xyFixed = ivec2(gl_WorkGroupID.xy);
    uint lid = gl_LocalInvocationIndex.x;

    // Skip pixel if stencil is empty
    if (texelFetch(stencil_texture, xyFixed, 0).x == 0u) {
      if (lid < 1) {
        imageStore(fields_texture, xyFixed, vec4(0));
      }
      return;
    } 

    // Map to domain pos
    vec2 domain_pos = (vec2(xyFixed) + vec2(0.5)) / vec2(texture_size);
    domain_pos = domain_pos * range + minBounds;

    // Iterate over points to obtain density/gradient
    vec3 v = vec3(0);
    for (uint i = lid; i < num_points; i += groupSize) {
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

  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

  layout(binding = 0, std430) restrict readonly buffer NodeBuffer { vec4 nodeBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer PosBuffer { vec2 posBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer DiamBuffer { vec4 diamBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;
  layout(binding = 0) uniform usampler2D stencilSampler;

  layout(location = 0) uniform uint nPos;     // nr of points
  layout(location = 1) uniform uint kNode;    // Node fanout
  layout(location = 2) uniform uint nLvls;    // Nr of tree levels
  layout(location = 3) uniform float theta;   // Approximation param
  layout(location = 4) uniform uvec2 textureSize;

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

  bool approx(vec2 domainPos, inout vec3 fieldValue) {
    // Fetch packed node data
    const vec4 vNode = nodeBuffer[loc];
    const vec4 vDiam = diamBuffer[loc];

    // Unpack node data
    const vec2 center = vNode.xy;
    const uint mass = uint(vNode.w);
    const vec2 diam = vDiam.xy;
    const uint begin = uint(vDiam.w);
    
    // Squared distance to pos
    vec2 t = domainPos - center;
    float t2 = dot(t, t);

    // Compute squared diameter
    vec2 b = abs(normalize(t)) * vec2(-1, 1);
    vec2 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);

      // Field layout is: S, V.x, V.y, V.z
      fieldValue += mass * vec3(tStud, t * (tStud * tStud));

      return true;
    } else if (mass <= 8 || lvl == nLvls - 1) {
      // Iterate over all leaf points (there goes thread divergence)
      for (uint i = begin; i < begin + mass; ++i) {
        t = domainPos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));

        // Field layout is: S, V.x, V.y, V.z
        fieldValue += vec3(tStud, t * (tStud * tStud));
       }
      return true;
    }
    return false;
  }

  vec3 traverse(vec2 domainPos) {
    vec3 fieldValue = vec3(0);

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
    const uvec2 globalIdx = gl_WorkGroupID.xy
                          * gl_WorkGroupSize.xy
                          + gl_LocalInvocationID.xy;
                   
    // Check that invocation is inside field texture
    if (min(globalIdx, textureSize - 1) != globalIdx) {
      return;
    }

    // Skip pixel if stencil is empty
    if (texelFetch(stencilSampler, ivec2(globalIdx), 0).x == 0u) {
      imageStore(fieldImage, ivec2(globalIdx), vec4(0));
      return;
    } 

    // Compute pixel position in [0, 1], then map to domain bounds
    vec2 domainPos = (vec2(globalIdx) + 0.5) / vec2(textureSize);
    domainPos = domainPos * bounds.range + bounds.min;

     // Traverse tree and store result
    vec3 v = traverse(domainPos);
    imageStore(fieldImage, ivec2(globalIdx), vec4(v, 0));
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

GLSL(point_bvh_src, 450,
  struct Bounds {
    vec2 min;
    vec2 max;
    vec2 range;
    vec2 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  layout(binding = 0, std430) restrict readonly buffer NodeBuffer { vec4 nodeBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer IdxBuffer { uint idxBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer PosBuffer { vec2 posBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer MinbBuffer { vec2 minbBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer DiamBuffer { vec2 diamBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 6, std430) restrict readonly buffer PosBuffer2 { vec2 posBufferUnsorted[]; };
  layout(binding = 7, std430) restrict writeonly buffer Val { vec3 Values[]; };
  layout(binding = 0) uniform usampler2D stencilSampler;

  layout(location = 0) uniform uint nPos;     // nr of points
  layout(location = 1) uniform uint kNode;    // Node fanout
  layout(location = 2) uniform uint nLvls;    // Nr of tree levels
  layout(location = 3) uniform float theta;   // Approximation param
  layout(location = 4) uniform uvec2 textureSize;

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

  bool approx(vec2 domainPos, inout vec3 fieldValue) {
    // Query values from current node
    vec4 node = nodeBuffer[loc];
    vec2 diam = diamBuffer[loc];
    
    // Squared distance to pos
    vec2 t = domainPos - node.xy;
    float t2 = dot(t, t);

    // Compute squared diameter
    vec2 b = abs(normalize(t)) * vec2(-1, 1);
    vec2 c = diam - b * dot(diam, b); // Vector rejection of diam onto unit vector b
    
    if (dot(c, c) / t2 < theta2 || node.w < 2.f) {
      // If BH-approximation passes, compute approximated value
      float tStud = 1.f / (1.f + t2);

      // Field layout is: S, V.x, V.y, V.z
      fieldValue += node.w * vec3(tStud, t * (tStud * tStud));

      return true;
    } else if (lvl == nLvls - 1) {
      // If a leaf node is reached that is not approximate enough
      // we iterate over all contained points (there goes thread divergence)
      uint begin = idxBuffer[loc];
      for (uint i = begin; i < begin + uint(node.w); ++i) {
        t = domainPos - posBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));

        // Field layout is: S, V.x, V.y, V.z
        fieldValue += vec3(tStud, t * (tStud * tStud));
       }
      return true;
    }
    return false;
  }

  vec3 traverse(vec2 domainPos) {
    vec3 fieldValue = vec3(0);

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
    const uint globalIdx = gl_WorkGroupID.x
                        * gl_WorkGroupSize.x
                        + gl_LocalInvocationID.x;
    if (min(globalIdx, nPos - 1) != globalIdx) {
      return;
    }

    vec2 domainPos = posBufferUnsorted[globalIdx];
    vec3 v = traverse(domainPos);
    Values[globalIdx] = v;

    // // Compute pixel position in [0, 1], then map to domain bounds
    // vec2 domainPos = (vec2(globalIdx) + 0.5) / vec2(textureSize);
    // domainPos = domainPos * bounds.range + bounds.min;

    //  // Traverse tree and store result
    // vec3 v = traverse(domainPos);
    // imageStore(fieldImage, ivec2(globalIdx), vec4(v, 0));
  }
);