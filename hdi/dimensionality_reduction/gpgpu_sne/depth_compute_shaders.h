#pragma once

#define GLSL_LOCAL_SIZE 128
#define GLSL(version, shader)  "#version " #version "\n" #shader

const char* depth_interpolation_src = GLSL(430,
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  const float bias = 0.0025f; // max accepted offset in depth map
  uniform uint num_points;
  uniform mat4 rotation;
  uniform sampler2D stencil_depth_texture;
  uniform sampler2D fields_texture;
  
  // vec3 removeZero(in vec3 v) {
  //   if (v.x == 0.f) v.x = 1.f;
  //   if (v.y == 0.f) v.y = 1.f;
  //   if (v.z == 0.f) v.z = 1.f;
  //   return v;
  // }

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;
    
    for (uint i = lid; i < num_points; i += groupSize) {
      // Position of point in range 0 to 1
      vec3 position = (Positions[i] - minBounds) * invRange;
      // Apply optional camera transformation
      position = vec3(rotation * vec4(position, 1));

      // Sample the input texture if measured depth is close
      float z = texture(stencil_depth_texture, position.xy).y;
      vec4 v = texture(fields_texture, position.xy);
      if (abs(position.z - z) < bias) {
        Values[i] = v;
      }
    }
  }
);