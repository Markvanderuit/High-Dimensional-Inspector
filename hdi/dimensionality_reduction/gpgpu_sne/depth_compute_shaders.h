#pragma once
#include "3d_compute_shaders.h"

GLSL(depth_sumq_src, 430,
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Val { vec4 Values[]; };
  layout(std430, binding = 1) buffer SumQ { float Sum[]; };

  uniform uint num_points;
  uniform uint iteration;
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared float reduction_array[halfGroupSize];

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    float sum_Q = 0.f;

    if (iteration == 0) {
      // First iteration adds all values
      for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
          i < num_points;
          i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        sum_Q += max(Values[i].x - 1.f, 0f);
      }
    } else if (iteration == 1) {
      // Second iteration adds resulting 128 values
      sum_Q = Sum[lid];      
    }

    // Reduce add to a single value
    if (lid >= halfGroupSize) {
      reduction_array[lid - halfGroupSize] = sum_Q;
    }
    barrier();
    if (lid < halfGroupSize) {
      reduction_array[lid] += sum_Q;
    }
    for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
      barrier();
      if (lid < i) {
        reduction_array[lid] += reduction_array[lid + i];
      }
    }
    barrier();
    if (lid < 1) {
      Sum[iteration > 0 ? 0 : gl_WorkGroupID.x] = reduction_array[0] + reduction_array[1];
    }
  }
);

GLSL(depth_bounds_src, 430,
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 positions[]; };
  layout(std430, binding = 1) buffer BoundsReduceAddInterface {
    vec3 minBoundsReduceAdd[128];
    vec3 maxBoundsReduceAdd[128];
  };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  uniform uint iteration;
  uniform uint num_points;
  uniform float padding;
  const uint halfGroupSize = gl_WorkGroupSize.x / 2;
  shared vec3 min_reduction[halfGroupSize];
  shared vec3 max_reduction[halfGroupSize];

  vec3 removeZero(in vec3 v) {
    if (v.x == 0.f) v.x = 1.f;
    if (v.y == 0.f) v.y = 1.f;
    if (v.z == 0.f) v.z = 1.f;
    return v;
  }

  void main() {
    const uint lid = gl_LocalInvocationIndex.x;
    vec3 min_local = vec3(1e38);
    vec3 max_local = vec3(-1e38);

    if (iteration == 0) {
      // First iteration adds all values
      for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
          i < num_points;
          i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        vec3 pos = positions[i];
        min_local = min(pos, min_local);
        max_local = max(pos, max_local);
      }
    } else if (iteration == 1) {
      // Second iteration adds resulting 128 values
      min_local = minBoundsReduceAdd[lid];
      max_local = maxBoundsReduceAdd[lid];
    }

    // Perform reduce-add over all points
    if (lid >= halfGroupSize) {
      min_reduction[lid - halfGroupSize] = min_local;
      max_reduction[lid - halfGroupSize] = max_local;
    }
    barrier();
    if (lid < halfGroupSize) {
      min_reduction[lid] = min(min_local, min_reduction[lid]);
      max_reduction[lid] = max(max_local, max_reduction[lid]);
    }
    for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
      barrier();
      if (lid < i) {
        min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + i]);
        max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + i]);
      }
    }
    barrier();
    // perform store in starting unit
    if (lid < 1) {
      min_local = min(min_reduction[0], min_reduction[1]);
      max_local = max(max_reduction[0], max_reduction[1]);
      if (iteration == 0) {
        minBoundsReduceAdd[gl_WorkGroupID.x] = min_local;
        maxBoundsReduceAdd[gl_WorkGroupID.x] = max_local;
      } else if (iteration == 1) {
        vec3 padding = (max_local - min_local) * 0.5f * padding;
        minBounds = min_local - padding;
        maxBounds = max_local + padding;
        range = (maxBounds - minBounds);
        invRange = 1.f / removeZero(range);
      }
    }
  }
);

GLSL(depth_3D_interpolation_src, 430,
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  // const float bias = 0.1f; // max accepted offset in depth map
  uniform uint num_points;
  uniform mat4 transformation;
  uniform sampler2DArray depth_textures;
  uniform sampler3D fields_texture;

  void main() {
    // Grid stride loop, straight from CUDA, scales better for very large N
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
         i < num_points;
         i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      // Map position of point to [0, 1]
      vec3 position = (Positions[i] - minBounds) * invRange;
      
      // Apply optional camera transformation
      position = vec3(transformation * vec4(position, 1));

      // Query the depth texture for z-bounds
      float zNear = texture(depth_textures, vec3(position.xy, 0)).x;
      float zFar = 1.f - texture(depth_textures, vec3(position.xy, 1)).x;

      // Map z-position to [zNear, zFar]
      position.z = (position.z - zNear) / (zFar - zNear);

      // Sample texture at position
      Values[i] = texture(fields_texture, position.xyz);
    }
  }
);


GLSL(depth_update_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer GradientLayout { vec3 Gradients[]; };
  layout(std430, binding = 2) buffer PrevGradientLayout { vec3 PrevGradients[]; };
  layout(std430, binding = 3) buffer GainLayout { vec3 Gain[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform float eta;
  uniform float minGain;
  uniform float mult;
  uniform float iter_mult;

  vec3 dir(vec3 v) {
    return vec3(
      v.x > 0 ? 1 : -1, 
      v.y > 0 ? 1 : -1, 
      v.z > 0 ? 1 : -1
    );
  }

  vec3 matches(vec3 a, vec3 b) {
    return vec3(
      sign(a.x) != sign(b.x) ? 1 : 0, 
      sign(a.y) != sign(b.y) ? 1 : 0, 
      sign(a.z) != sign(b.z) ? 1 : 0
    );
  }

  void main() {
    // Grid stride loop, straight from CUDA, scales better for very large N
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
         i < num_points;
         i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      vec3 grad = Gradients[i];
      vec3 pgrad = PrevGradients[i];
      vec3 gain = Gain[i];

      vec3 match = matches(grad, pgrad);
      gain = match * (gain + 0.2) + (1 - match) * (gain * 0.8);
      gain = max(gain, vec3(minGain));

      vec3 etaGain = eta * gain;
      grad = dir(grad) * abs(grad * etaGain) / etaGain;
      pgrad = iter_mult * pgrad - etaGain * grad;

      Gain[i] = gain;
      PrevGradients[i] = pgrad;
      Positions[i] += pgrad * mult;
    }
  }
);

GLSL(depth_centering_src, 430,
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  uniform uint num_points;
  uniform vec3 center;
  uniform float scaling;

  void main() {
    // Grid stride loop, straight from CUDA, scales better for very large N
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
          i < num_points;
          i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      Positions[i] = scaling * (Positions[i] - center);
    }    
  }
);

GLSL(depth_positive_forces_src, 430,
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Neigh { uint Neighbours[]; };
  layout(std430, binding = 2) buffer Prob { float Probabilities[]; };
  layout(std430, binding = 3) buffer Ind { int Indices[]; };
  layout(std430, binding = 4) buffer Posit { vec3 PositiveForces[]; };

  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = gl_WorkGroupSize.x / 2;
  uniform uint num_points;
  uniform float inv_num_points;
  shared vec3 reduction_array[halfGroupSize];

  void main () {
    const uint i = gl_WorkGroupID.x;
    const uint lid = gl_LocalInvocationID.x;
    if (i >= num_points) {
      return;
    }

    // Get the point coordinates
    vec3 point_i = Positions[i];

    // Computing positive forces using neighbors
    int index = Indices[i * 2 + 0];
    int size = Indices[i * 2 + 1];
    vec3 positive_force = vec3(0);
    for (uint j = lid; j < size; j += groupSize) {
      // Get other point coordinates
      vec3 point_j = Positions[Neighbours[index + j]];

      // Calculate 2D distance between the two points
      vec3 dist = point_i - point_j;

      // Similarity measure of the two points
      float qij = 1.f / (1.f + dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

      // Calculate the attractive force
      positive_force += Probabilities[index + j] * qij * dist;
    }
    positive_force *= inv_num_points;

    // Reduce add sum_positive_red to a single value
    if (lid >= halfGroupSize) {
      reduction_array[lid - halfGroupSize] = positive_force;
    }
    barrier();
    if (lid < halfGroupSize) {
      reduction_array[lid] += positive_force;
    }
    for (uint reduceSize = halfGroupSize / 2; reduceSize > 1; reduceSize /= 2) {
      barrier();
      if (lid < reduceSize) {
        reduction_array[lid] += reduction_array[lid + reduceSize];
      }
    }
    barrier();
    if (lid < 1) {
      PositiveForces[i] = reduction_array[0] + reduction_array[1];
    } 
  }
);

GLSL(depth_gradients_src, 430,
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Posit { vec3 PositiveForces[]; };
  layout(std430, binding = 1) buffer Fiel { vec4 Fields[]; };
  layout(std430, binding = 2) buffer Grad { vec3 Gradients[]; };

  uniform uint num_points;
  uniform float exaggeration;
  uniform float sum_Q;
  uniform float inv_sum_Q;

  void main() {
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
          i < num_points;
          i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      vec3 sum_positive = PositiveForces[i];
      vec3 sum_negative = Fields[i].yzw * inv_sum_Q;
      Gradients[i] = 4.f * (exaggeration * sum_positive - sum_negative);
    }
  }
);