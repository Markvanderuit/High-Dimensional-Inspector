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

namespace hdi::dbg::_3d {
  GLSL(bvh_vert, 450,
    // Wrapper structure for BoundsBuffer data
    struct Bounds {
      vec3 min;
      vec3 max;
      vec3 range;
      vec3 invRange;
    };

    layout (location = 0) in vec3 vert; // instanced
    layout (location = 1) in vec3 minb;
    layout (location = 2) in vec3 diam;
    layout (location = 3) in float flag;
    layout (location = 0) out vec3 pos;
    layout (location = 1) out float theta;
    layout (location = 2) out float fflag;

    layout (location = 0) uniform mat4 uTransform;
    layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

    void main() {
      fflag = flag;

      // Obtain normalized [0, 1] boundary values
      vec3 _minb = (minb - bounds.min) * bounds.invRange;
      vec3 _diam = diam * bounds.invRange;

      // Generate position from instanced data
      pos = _minb + (0.5 + vert) * _diam;

      // Compute uhh, anisotropy of box
      theta = dot(normalize(_diam), normalize(vec3(1)));

      // Apply camera transformation to output data
      gl_Position = uTransform * vec4(pos, 1);
    }
  );

  GLSL(bvh_frag, 450,
    layout (location = 0) in vec3 pos;
    layout (location = 1) in float theta;
    layout (location = 2) in float flag;
    layout (location = 0) out vec4 color;
    layout (location = 1) uniform float uCubeOpacity;
    layout (location = 2) uniform float uBarnesHutOpacity;
    layout (location = 3) uniform float uTheta;
    
    void main() {
      if (uBarnesHutOpacity > 0.f && flag > 0.f) {
        if (theta < uTheta) {
          color =  vec4(1, 0, 0, 1);
        } else {
          color =  vec4(0, 0, 0, uBarnesHutOpacity);
        }
      } else if (uCubeOpacity > 0.f) {
        if (theta < uTheta) {
          color =  vec4(1, 0, 0, 1);
        } else {
          color =  vec4(normalize(pos), uCubeOpacity);
        }
      } else {
        discard;
      }
    }
  );
}