#pragma once
#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

GLSL(triangulate_comp, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  layout (binding = 0, std430) restrict readonly buffer MinBBuffer { vec3 minBBuffer[]; };
  layout (binding = 1, std430) restrict readonly buffer MaxBBuffer { vec3 maxBBuffer[]; };
  layout (binding = 2, std430) restrict readonly buffer MassBuffer { uint massBuffer[]; };
  layout (binding = 3, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout (binding = 4, std430) restrict writeonly buffer VertexBuffer { vec4 vertexBuffer[]; };
  layout (location = 0) uniform uint nNode; // Nr. of nodes to process
  layout (location = 1) uniform uint nPos;  // Nr. of pos in embedding

  void main() {
    // Invocation ID
    const uint globalIdx = gl_WorkGroupID.x
                         * gl_WorkGroupSize.x
                         + gl_LocalInvocationID.x;

    // Check that invocation is inside valid range
    if (min(globalIdx, nNode - 1) != globalIdx) {
      return;
    }

    // Load in node values
    float mass = float(massBuffer[globalIdx]);
    float density = 1.f - (mass / float(nPos));
    vec4 minb = vec4(minBBuffer[globalIdx], density);
    vec3 maxb = maxBBuffer[globalIdx];
    vec3 add = maxb - minb.xyz;

    // Oh boy
    const uint outIdx = 24 * globalIdx;
    vertexBuffer[outIdx + 0] = minb + vec4(0, 0, 0, 0);
    vertexBuffer[outIdx + 1] = minb + vec4(add.x, 0, 0, 0);
    vertexBuffer[outIdx + 2] = minb + vec4(add.x, 0, 0, 0);
    vertexBuffer[outIdx + 3] = minb + vec4(add.x, add.y, 0, 0);
    vertexBuffer[outIdx + 4] = minb + vec4(add.x, add.y, 0, 0);
    vertexBuffer[outIdx + 5] = minb + vec4(0, add.y, 0, 0);
    vertexBuffer[outIdx + 6] = minb + vec4(0, add.y, 0, 0);
    vertexBuffer[outIdx + 7] = minb + vec4(0, 0, 0, 0);
    vertexBuffer[outIdx + 8] = minb + vec4(0, 0, add.z, 0);
    vertexBuffer[outIdx + 9] = minb + vec4(add.x, 0, add.z, 0);
    vertexBuffer[outIdx + 10] = minb + vec4(add.x, 0, add.z, 0);
    vertexBuffer[outIdx + 11] = minb + vec4(add.x, add.y, add.z, 0);
    vertexBuffer[outIdx + 12] = minb + vec4(add.x, add.y, add.z, 0);
    vertexBuffer[outIdx + 13] = minb + vec4(0, add.y, add.z, 0);
    vertexBuffer[outIdx + 14] = minb + vec4(0, add.y, add.z, 0);
    vertexBuffer[outIdx + 15] = minb + vec4(0, 0, add.z, 0);
    vertexBuffer[outIdx + 16] = minb + vec4(0, 0, 0, 0);
    vertexBuffer[outIdx + 17] = minb + vec4(0, 0, add.z, 0);
    vertexBuffer[outIdx + 18] = minb + vec4(add.x, 0, 0, 0);
    vertexBuffer[outIdx + 19] = minb + vec4(add.x, 0, add.z, 0);
    vertexBuffer[outIdx + 20] = minb + vec4(add.x, add.y, 0, 0);
    vertexBuffer[outIdx + 21] = minb + vec4(add.x, add.y, add.z, 0);
    vertexBuffer[outIdx + 22] = minb + vec4(0, add.y, 0, 0);
    vertexBuffer[outIdx + 23] = minb + vec4(0, add.y, add.z, 0);
  }
);

GLSL(draw_bvh_vert, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout (location = 0) in vec4 vertex;
  layout (location = 0) out float density;
  layout (location = 0) uniform mat4 uTransform;
  layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

  void main() {
    vec3 pos = (vertex.xyz - bounds.min) * bounds.invRange;
    density = vertex.w;
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(draw_bvh_frag, 450,
  layout (location = 0) in float density;
  layout (location = 0) out vec4 color;

  void main() {
    // vec3 m = mix(vec3(1, 0, 1), vec3(0, 1, 1), density);
    color = vec4(vec3(0, 1, 1), 0.1);
  }
);

GLSL(draw_emb_vert, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout (location = 0) in vec4 vertex;
  layout (location = 0) out vec3 pos;
  layout (location = 0) uniform mat4 uTransform;
  layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

  void main() {
    pos = (vertex.xyz - bounds.min) * bounds.invRange;
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(draw_emb_frag, 450,
  layout (location = 0) in vec3 pos;
  layout (location = 0) out vec4 color;

  void main() {
    color = vec4(pos, 1.0);
    // color = vec4(0.2, 1, 0.2, 1.0);
  }
);