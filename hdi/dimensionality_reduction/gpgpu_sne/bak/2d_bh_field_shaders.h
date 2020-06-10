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

/**
 * Compute shader for computation of a stencil texture.
 * 
 * Computes a stencil texture which applies a 3x3 feathering kernel to
 * all points in the lowest level of the barnes-hut tree.
 */
GLSL(stencil_src, 430,
  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
  layout(std430, binding = 0) buffer AtlasDataBlock { 
    struct {
      uvec2 offset; 
      uvec2 size;
    } atlasLevelData[];
  };
  layout(r8ui, binding = 0) writeonly uniform uimage2D stencil_texture;
  
  uniform sampler2D densityAtlasSampler;
  uniform uint level;

  // Shared memory into which we load large part of the texture
  const ivec2 groupSize = ivec2(gl_WorkGroupSize);
  const ivec2 atlasValueBorder = ivec2(1); // 1px border padding
  const ivec2 atlasValueDims = groupSize + 2 * atlasValueBorder;
  shared float sAtlasValues[atlasValueDims.x * atlasValueDims.y];

  // Kernel offsets
  const uint nOffsets = 9u;
  const ivec2[] offsets = ivec2[](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
    ivec2(-1, 0), ivec2(0, 0), ivec2(1, 0),
    ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1)
  );

  /**
   * Obtain correct index for reading shared values (sAtlasValues).
   */
  int sAtlasIdx(ivec2 lxy) {
    return atlasValueDims.x * lxy.y + lxy.x;
  }

  void main() {
    const ivec2 xy = ivec2(gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID);
    const ivec2 lxy = ivec2(gl_LocalInvocationID);
 
    // Atlas texel offsets for current layer
    const ivec2 atlasOffset = ivec2(atlasLevelData[level].offset);

    // Fetch texels at current invocation locations
    const ivec2 _xy = xy - atlasValueBorder + atlasOffset;
    sAtlasValues[sAtlasIdx(lxy)] = texelFetch(densityAtlasSampler, _xy, 0).x;

    // Fetch texels at horizontal borders
    if (lxy.x < 2) { 
      sAtlasValues[sAtlasIdx(lxy + ivec2(groupSize.x, 0))] 
        = texelFetch(densityAtlasSampler, _xy + ivec2(groupSize.x, 0), 0).x;
    }

    // Fetch texels at vertical borders
    if (lxy.y < 2) { 
      sAtlasValues[sAtlasIdx(lxy + ivec2(0, groupSize.y))] 
        = texelFetch(densityAtlasSampler, _xy + ivec2(0, groupSize.y), 0).x;
    }

    // Fetch texels at corners
    if (lessThan(lxy, ivec2(2)) == bvec2(true)) {
      sAtlasValues[sAtlasIdx(lxy + groupSize)] 
        = texelFetch(densityAtlasSampler, _xy + groupSize, 0).x;
    }

    // Synchronize shared memory across group
    barrier();

    // Sum values inside kernel, reading from shared memory only
    float stencilValue = 0.f;
    const ivec2 _lxy = atlasValueBorder + lxy;
    for (int i = 0; i < nOffsets; i++) {
      stencilValue += sAtlasValues[sAtlasIdx(_lxy + offsets[i])];
    }
    
    // Store result in stencil texture
    imageStore(stencil_texture, xy, uvec4(uint(stencilValue > 0.f)));
  }
);

/**
 * Compute shader for field computation using Barnes-Hut approximation.
 * 
 * Per field pixel, computes density and gradient values in O(Px log Px)
 * upper time bound, by performing a DFS over a quadtree.
 */
GLSL(tree_traverse_src, 430,
  // 16x16=256 invocations per work group
  layout(local_size_x = 16, 
         local_size_y = 16, 
         local_size_z = 1) in;
  layout(std430, binding = 0) buffer BoundsBuffer {
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };
  layout(std430, binding = 1) buffer AtlasBuffer {
    struct {
      uvec2 offset; 
      uvec2 size;
    } atlasLevelData[];
  };
  layout(rgba32f, binding = 0) writeonly uniform image2D fieldTexture;

  uniform uint nPoints;
  uniform uint nLevels;
  uniform float theta2;
  uniform ivec2 fieldTextureSize;
  uniform ivec2 densityAtlasSize;
  uniform usampler2D stencilSampler;
  uniform sampler2D densityAtlasSampler;

  // Shared atlas layout data
  shared ivec2 sAtlasOffsets[12];
  shared ivec2 sAtlasSizes[12];
  shared float sAtlasDiams2[12];

  /**
   * Query the atlas for specific node data at the current location and level.
   */
  vec3 queryAtlas(ivec2 loc, uint lvl) {
    ivec2 _loc = sAtlasOffsets[lvl] + loc;
    return texelFetch(densityAtlasSampler, _loc, 0).xyz;
  }

  /**
   * Simple structure for DFS tree traversal.
   * 
   * Stores necessary data for stack-based traversal, such as the
   * current node's location, tree depth, and a stack for visited
   * nodes.
   */
  struct Traversal {
    ivec2 loc;
    uint stack;
    uint depth;
  };

  /**
   * Perform a tree descent for current traversal data.
   * 
   * Shifts location into tree depth, and pushes newly unvisited locations
   * on the stack.
   */
  void descend(inout Traversal trv) {
    // Increase depth and shift location to "descend"
    trv.depth++;
    trv.loc <<= 1;

    // Push newly unvisited locations on stack
    trv.stack |= 3u << (2 * trv.depth);
  }

  /**
   * Perform a tree ascent for current traversal data
   * 
   * Shifts location upwards or forward in tree depth, and pops currently 
   * visited location off the stack.
   */
  void ascend(inout Traversal trv) {
    // Ascend if there is no location left to visit on this level
    uint nextDepth = findMSB(trv.stack) / 2;
    trv.loc >>= (trv.depth - nextDepth);

    // Pop old visited location from stack
    uint shift = 2 * nextDepth;
    uint b = (trv.stack >> shift) - 1;
    trv.stack &= ~(3u << shift);
    trv.stack |= b << shift;

    // Move to next location
    int lMod = trv.loc.x % 2;
    trv.loc += ivec2(1 - 2 * lMod, lMod);
    trv.depth = nextDepth;
  }
  

  /**
   * Test tree node for Barnes-Hut approximation.
   * 
   * Test the current tree node, checking if it is
   * (a) a leaf, or
   * (b) approximate enough approximation, given the domainPos position, s.t.
   * 
   * in which cases we compute field values using this node, and the
   * traversal is told to stop descending along the current path, i.e. we
   * truncate the tree at this node.
   */
  bool approx(Traversal trv, inout vec3 field, vec2 domainPos) {
    // Query atlas at current location and level
    uint lvl = (nLevels - 1) - trv.depth; 
    vec3 query = queryAtlas(trv.loc, lvl);
    
    // Squared euclid. distance between pixel and node center of mass
    vec2 t = domainPos - query.yz;
    float t_2 = dot(t, t);

    // Do we truncate the tree at this node?
    bool truncate =
      (query.x <= 1.f) || (lvl == 0u) || // is leaf
      (sAtlasDiams2[lvl] / t_2) < theta2; // is approximation
    
    // Accumulate field value
    if (truncate) {
      float t_stud = 1.f / (1.f + t_2);
      vec2 t_stud_2 = t * (t_stud * t_stud);
      field += query.x * vec3(t_stud, t_stud_2);
    }

    return truncate;
  }
  
  /**
   * Compute field positional value through tree traversal.
   * 
   * Performs Barnes-Hut approximation through a stack-based DFS
   * tree traversal, approximating field values for domainPos through
   * comparison with the tree's nodes instead of points.
   */
  vec3 bhTraverse(vec2 domainPos) {
    vec3 v = vec3(0);

    Traversal trv = Traversal(ivec2(0), 0u, 0u);
    do {
      if (approx(trv, v, domainPos)) {
        ascend(trv);
      } else {
        descend(trv);
      }
    } while (trv.depth != 0u);

    return v;
  }

  void main() {
    // Location of current invocation
    const ivec2 xy = ivec2(gl_WorkGroupSize * gl_WorkGroupID + gl_LocalInvocationID);
    const ivec2 lxy = ivec2(gl_LocalInvocationID);
    const uint lid = gl_LocalInvocationIndex;

    // Load atlas layout data into shared memory
    float rPixel = pow(length(vec2(range) / vec2(fieldTextureSize)), 2);
    if (lid < nLevels) {
      sAtlasOffsets[lid] = ivec2(atlasLevelData[lid].offset);
      sAtlasSizes[lid] = ivec2(atlasLevelData[lid].size);
      float rBbox = pow(length(vec2(range) / vec2(sAtlasSizes[lid])), 2);
      sAtlasDiams2[lid] = max(rPixel, rBbox); // Diam^2 for BH condition
    }

    // Synchronize load of atlas layout
    barrier();

    // Invocations should only compute for valid pixels
    if (min(xy, fieldTextureSize - 1) != xy) {
      return;
    }

    // Location of current invocation in embedding domain
    vec2 dxy = (vec2(xy) + vec2(0.5)) / vec2(fieldTextureSize);
    dxy = dxy * range + minBounds;

    // Invocations should only compute for non-empty pixels
    vec3 v = vec3(0);
    // if (texelFetch(stencilSampler, xy, 0).x != 0u) {
    v = bhTraverse(dxy);
    // } 

    // Perform tree traversal and store result
    // vec3 v = bhTraverse(dxy);
    imageStore(fieldTexture, xy, vec4(v, 0));
  }
);

// Compute shader for point sampling from field
GLSL(interp_src, 430,
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec3 Values[]; };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec2 minBounds;
    vec2 maxBounds;
    vec2 range;
    vec2 invRange;
  };

  uniform uint num_points;
  uniform uvec2 texture_size;
  uniform sampler2D fields_texture;

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