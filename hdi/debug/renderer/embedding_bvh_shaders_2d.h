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

namespace hdi::dbg::_2d {
  GLSL(bvh_vert, 450,
    // Wrapper structure for BoundsBuffer data
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };
    
    // Ten nice colors for color mapping
    const vec3 labels[10] = vec3[10](
      vec3(16, 78, 139),
      vec3(139, 90, 43),
      vec3(138, 43, 226),
      vec3(0, 128, 0),
      vec3(255, 150, 0),
      vec3(204, 40, 40),
      vec3(131, 139, 131),
      vec3(0, 205, 0),
      vec3(20, 20, 20),
      vec3(0, 150, 255)
    );

    layout(location = 0) in vec2 vert; // instanced
    layout(location = 1) in vec2 minb;
    layout(location = 2) in vec4 node0;
    layout(location = 3) in vec4 node1;
    layout(location = 4) in uint flag;

    layout(location = 0) out vec2 posOut;
    layout(location = 1) out vec3 colorOut;
    // layout(location = 1) out float fflag;

    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform float uCubeOpacity;
    layout(location = 2) uniform uint lvl;
    layout(location = 3) uniform bool doLvl;
    layout(location = 4) uniform bool doFlags;

    layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

    void main() {
      // Calculate range in which instance should fall
      uint nBegin = 0;
      for (uint i = 0; i < lvl; ++i) {
        nBegin |= 1u << (BVH_LOGK_2D * i);
      }
      uint nEnd = nBegin | (1u << (BVH_LOGK_2D * (lvl)));

      // Obtain normalized [0, 1] boundary values
      vec2 _minb = (minb - bounds.min) * bounds.invRange;
      vec2 _diam = node1.xy * bounds.invRange;

      // Embedding node leaves have zero area, account for this in visualization
      if (_diam == vec2(0.f)) {
        _diam = vec2(0.005f);
        _minb = _minb - 0.5f * _diam;
      }
      
      // Generate position from instanced data
      if (node0.w == 0 || (doFlags && flag == 0) || (doLvl && (gl_InstanceID < nBegin || gl_InstanceID >= nEnd))) {
        posOut = vec2(-999); 
      } else {
        posOut = _minb + (0.5f + vert) * _diam;
        posOut.y = 1.f - posOut.y;
      }

      // Output color
      colorOut = labels[doFlags ? 0 : gl_InstanceID % 10] / 255.f;

      // Apply camera transformation to output data
      gl_Position = uTransform * vec4(posOut, 0, 1);
    }
  );

  GLSL(bvh_frag, 450,
    layout(location = 0) in vec2 posIn;
    layout(location = 1) in vec3 colorIn;

    layout(location = 0) out vec4 colorOut;

    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform float uCubeOpacity;
    layout(location = 2) uniform uint lvl;
    layout(location = 3) uniform bool doLvl;
    
    void main() {
      if (uCubeOpacity > 0.f) {
        colorOut =  vec4(colorIn, uCubeOpacity);
      } else {
        discard;
      }
    }
  );
}