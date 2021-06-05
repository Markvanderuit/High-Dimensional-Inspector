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
  GLSL(vert_quad, 450,
    // Inputs
    layout(location = 0) in vec2 vertexIn;
    layout(location = 1) in vec3 posIn;
    layout(location = 2) in uint labelIn;

    // Outputs
    layout(location = 0) out vec3 fragOut;
    layout(location = 1) out vec3 posOut;
    layout(location = 2) out vec4 colorOut;

    // Uniform values
    layout(location = 0) uniform mat4 transform;
    layout(location = 1) uniform bool drawLabels;
    layout(location = 2) uniform float size;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { 
      struct {
        vec3 min;
        vec3 max;
        vec3 range;
        vec3 invRange;
      } bounds;
    };

    // Label colors (10x)
    const vec3 labels[10] = vec3[10](
      vec3(16, 78, 139),
      vec3(139, 90, 43),
      vec3(138, 43, 226),
      vec3(0, 128, 0),
      vec3(255, 150, 0),
      vec3(204, 40, 40),
      vec3(131, 139, 131),
      vec3(0, 205, 0),
      vec3(235, 235, 235),
      // vec3(20, 20, 20),
      vec3(0, 150, 255)
    );

    void main() {
      // Output vertex position
      // 1. Transform embedding position to [0, 1] space
      // 2. Then offset by vertex position
      // 3. Then apply transformation matrix
      posOut = (posIn - bounds.min) * bounds.invRange;
      fragOut = posOut + vec3(vertexIn, 0) * size;
      gl_Position = transform * vec4(posOut, 1) + vec4(vertexIn, 0, 0) * size;

      // Output label color
      colorOut = vec4(labels[drawLabels ? labelIn % 10 : 0] / 255.f, 1);       
    }
  );

  GLSL(frag_quad, 450,
    // Inputs
    layout(location = 0) in vec3 fragIn;
    layout(location = 1) in vec3 posIn;
    layout(location = 2) in vec4 colorIn;

    // Outputs
    layout(location = 0) out vec4 colorOut;

    // Uniform values
    layout(location = 0) uniform mat4 transform;
    layout(location = 1) uniform bool drawLabels;
    layout(location = 2) uniform float size;

    void main() {
      const float r = 0.5 * size;
      const float t = distance(fragIn, posIn);

      // Discard fragment outside of circle radius
      if (t > r) {
        discard;
      }

      // Pass through color value
      colorOut = colorIn;
    }
  );
}