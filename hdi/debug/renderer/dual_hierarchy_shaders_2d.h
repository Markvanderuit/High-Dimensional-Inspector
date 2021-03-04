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
  /* GLSL(dot_vert, 450,
    layout(location = 0) in vec2 vertIn;
    layout(location = 0) uniform mat4 transform;

    void main() {
      gl_Position = transform * vec4(vertIn, 0, 1);
    }
  );

  GLSL(dot_frag, 450,
    layout(location = 0) out vec4 colorOut;
    
    void main() {
      colorOut = vec4(0, 0, 0, 1);
    }
  ); */

  GLSL(lines_vert_src, 450,
    // Wrapper structure for Bounds buffer data
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    // Input attributes
    layout(location = 0) in vec2 vert;
    layout(location = 1) in uvec2 pair;  // { pair.f, pair.e } -> { pair.x, pair.y }

    // Output attributes
    layout(location = 0) out vec4 colorOut;

    // Uniforms
    layout(location = 0) uniform mat4 transform;
    layout(location = 1) uniform uint lvl;
    layout(location = 2) uniform vec2 mousepos;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(binding = 1, std430) restrict readonly buffer FieldBuffer { vec3 fieldBuffer[]; };
    layout(binding = 2, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
    layout(binding = 3, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
    
    uint shrinkBits15(uint i) {
      i = i & 0x55555555u;
      i = (i | (i >> 1u)) & 0x33333333u;
      i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
      i = (i | (i >> 4u)) & 0x00FF00FFu;
      i = (i | (i >> 8u)) & 0x0000FFFFu;
      return i;
    }

    uvec2 decode(uint i) {
      uint x = shrinkBits15(i);
      uint y = shrinkBits15(i >> 1);
      return uvec2(x, y);
    }

    void main() {
      // Determine lvl of node
      uint fLvl = 0;
      {
        uint f = pair.x;
        do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_2D) > 0);
      }
      
      // Compute field node data implicitly
      const uint offset = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl));
      const uvec2 px = decode(pair.x - offset);
      const vec2 fbbox = vec2(1.f) / vec2(float(1u << fLvl));
      const vec2 fpos = fbbox * (vec2(px) + 0.5f);

      // Test whether node data is level/position relevant
      if (fLvl != lvl || clamp(mousepos, fpos - 0.5 * fbbox, fpos + 0.5 * fbbox) != mousepos) {
        colorOut = vec4(0);
        gl_Position = vec4(0);
        return;
      }

      const bool isFieldNode = gl_VertexID % 2 == 0;
      vec2 posOut;
      if (isFieldNode) {
        posOut = fpos;
      } else {
        // Obtain embedding node data from buffers
        const vec4 eNode0 = eNode0Buffer[pair.y];
        posOut = (eNode0.xy - bounds.min) * bounds.invRange;
      }
      posOut.y = 1.f - posOut.y;

      // Output color
      colorOut = vec4(1, 0, 1, 1);
      /* vec3 field = fieldBuffer[pair.x];
      if (field != vec3(0)) {
        colorOut = vec4(0.5 + 0.5 * normalize(field.yz), 0.5, 1);
      } else {
        colorOut = vec4(0, 0, 0, 1); // vec4(0.5, 0.5, 0.5, 1);
      } */

      // Apply camera transformation to output data
      gl_Position = transform * vec4(posOut, 0, 1);
    }
  );

  GLSL(lines_frag_src, 450,
    layout(location = 0) in vec4 colorIn;
    layout(location = 0) out vec4 colorOut;

    void main() {
      if (colorIn.w == 0.f) {
        discard;
      }

      colorOut = colorIn;
    }
  );

  GLSL(nodes_vert_src, 450,
    // Wrapper structure for Bounds buffer data
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    // Input attributes
    layout(location = 0) in vec2 vert;
    layout(location = 1) in uvec2 pair;  // { pair.f, pair.e } -> { pair.x, pair.y }

    // Output attributes
    layout(location = 0) out vec4 colorOut;

    // Uniforms
    layout(location = 0) uniform mat4 transform;
    layout(location = 1) uniform uint lvl;
    layout(location = 2) uniform vec2 mousepos;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(binding = 1, std430) restrict readonly buffer FieldBuffer { vec3 fieldBuffer[]; };
    layout(binding = 2, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
    layout(binding = 3, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };

    uint shrinkBits15(uint i) {
      i = i & 0x55555555u;
      i = (i | (i >> 1u)) & 0x33333333u;
      i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
      i = (i | (i >> 4u)) & 0x00FF00FFu;
      i = (i | (i >> 8u)) & 0x0000FFFFu;
      return i;
    }

    uvec2 decode(uint i) {
      uint x = shrinkBits15(i);
      uint y = shrinkBits15(i >> 1);
      return uvec2(x, y);
    }

    void main() {
      // Determine lvl of node
      uint fLvl = 0;
      {
        uint f = pair.x;
        do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_2D) > 0);
      }
      
      // Compute field node data implicitly
      const uint offset = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl));
      const uvec2 px = decode(pair.x - offset);
      const vec2 fbbox = vec2(1.f) / vec2(float(1u << fLvl));
      const vec2 fpos = fbbox * (vec2(px) + 0.5f);

      // Test whether node data is level/position relevant
      if (fLvl != lvl || clamp(mousepos, fpos - 0.5 * fbbox, fpos + 0.5 * fbbox) != mousepos) {
        colorOut = vec4(0);
        gl_Position = vec4(0);
        return;
      }

      const bool isFieldNode = gl_VertexID < 8; // first 4 vertices draw field node, rest embedding
      vec2 posOut;
      if (isFieldNode) {
        posOut = fpos + vert * fbbox;
      } else {
        // Obtain embedding node data from buffers
        const vec4 eNode0 = eNode0Buffer[pair.y];
        const vec4 eNode1 = eNode1Buffer[pair.y];
        const vec2 epos = (eNode0.xy - bounds.min) * bounds.invRange;
        const vec2 ebbox = eNode1.xy * bounds.invRange;
        posOut = epos + vert * ebbox;
      }
      posOut.y = 1.f - posOut.y;

      // Output color
      if (isFieldNode) {
        colorOut = vec4(1, 0, 0, 1);
      } else {
        colorOut = vec4(0, 0, 1, 1);
      }

      // Apply camera transformation to output data
      gl_Position = transform * vec4(posOut, 0, 1);
    }
  );

  GLSL(nodes_frag_src, 450,
    layout(location = 0) in vec4 colorIn;
    layout(location = 0) out vec4 colorOut;

    void main() {
      if (colorIn.w == 0.f) {
        discard;
      }

      colorOut = colorIn;
    }
  );
} // namespace hdi::dbg::_2d