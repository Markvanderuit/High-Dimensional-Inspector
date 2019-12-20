#pragma once
#define GLSL_LOCAL_SIZE 128
#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

GLSL(sumq_src, 430,
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Val { vec4 Values[]; };
  layout(std430, binding = 1) buffer SumQ { float Sum[]; };

  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared float reduction_array[halfGroupSize];
  uniform uint num_points;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;

    // Compute sum_Q over part of points
    float sum_Q = 0.f;
    for (uint i = lid; i < num_points; i += groupSize) {
      sum_Q += max(Values[i].x - 1.f, 0f);
    }

    // Reduce add sum_Q to a single value
    if (lid >= halfGroupSize) {
      reduction_array[lid - halfGroupSize] = sum_Q;
    }
    barrier();
    if (lid < 64) {
      reduction_array[lid] += sum_Q;
    }
    for (uint reduceSize = halfGroupSize / 2; reduceSize > 1; reduceSize /= 2) {
      barrier();
      if (lid < reduceSize) {
        reduction_array[lid] += reduction_array[lid + reduceSize];
      }
    }
    barrier();
    if (lid < 1) {
      Sum[0] = reduction_array[0] + reduction_array[1];
    }
  }
);

// Copied from compute_shaders.glsl, adapted for 3d
GLSL(interpolation_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  shared float reduction_array[64];
  uniform uint num_points;
  uniform sampler3D fields;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;

    for (uint i = lid; i < num_points; i += groupSize) {
      // Position of point in range 0 to 1
      vec3 position = (Positions[i] - minBounds) * invRange;

      // Trilinearly sample the input texture
      vec4 v = texture(fields, position);
      Values[i] = v;
    }
  }
);

// Copied from compute_shaders.glsl, adapted for 3d
GLSL(forces_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Neigh { uint Neighbours[]; };
  layout(std430, binding = 2) buffer Prob { float Probabilities[]; };
  layout(std430, binding = 3) buffer Ind { int Indices[]; };
  layout(std430, binding = 4) buffer Fiel { vec4 Fields[]; };
  layout(std430, binding = 5) buffer Grad { vec3 Gradients[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = gl_WorkGroupSize.x / 2;
  shared vec3 sum_positive_red[halfGroupSize];
  uniform uint num_points;
  uniform float exaggeration;
  uniform float sum_Q;

  void main() {
    uint i = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;

    float inv_num_points = 1.0 / float(num_points);
    float inv_sum_Q = 1.0 / sum_Q;

    if (i >= num_points) {
      return;
    }

    // Get the point coordinates
    vec3 point_i = Positions[i];

    // Computing positive forces
    vec3 sum_positive = vec3(0);

    int index = Indices[i * 2 + 0];
    int size = Indices[i * 2 + 1];

    vec3 positive_force = vec3(0);
    for (uint j = lid; j < size; j += groupSize) {
      // Get other point coordinates
      vec3 point_j = Positions[Neighbours[index + j]];

      // Calculate 2D distance between the two points
      vec3 dist = point_i - point_j;

      // Similarity measure of the two points
      float qij = 1 / (1 + dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

      // Calculate the attractive force
      positive_force += Probabilities[index + j] * qij * dist * inv_num_points;
    }

    // Reduce add sum_positive_red to a single value
    if (lid >= halfGroupSize) {
      sum_positive_red[lid - halfGroupSize] = positive_force;
    }
    barrier();
    if (lid < halfGroupSize) {
      sum_positive_red[lid] += positive_force;
    }
    for (uint reduceSize = halfGroupSize / 2; reduceSize > 1; reduceSize /= 2) {
      barrier();
      if (lid < reduceSize) {
        sum_positive_red[lid] += sum_positive_red[lid + reduceSize];
      }
    }
    barrier();
    if (lid < 1) {
      sum_positive = sum_positive_red[0] + sum_positive_red[1];
      vec3 sum_negative = Fields[i].yzw * inv_sum_Q;
      Gradients[i] = 4 * (exaggeration * sum_positive - sum_negative);
    }
  }
);

// Copied from compute_shaders.glsl, adapted for 3d
GLSL(update_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer GradientLayout { vec3 Gradients[]; };
  layout(std430, binding = 2) buffer PrevGradientLayout { vec3 PrevGradients[]; };
  layout(std430, binding = 3) buffer GainLayout { vec3 Gain[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

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
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points) {
      return;
    }

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
);

// Copied from compute_shaders.glsl, adapted for 3d
GLSL(bounds_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform float padding;

  const uint workgroupSize = gl_WorkGroupSize.x;
  const uint halfWorkgroupSize = gl_WorkGroupSize.x / 2;

  shared vec3 min_reduction[halfWorkgroupSize];
  shared vec3 max_reduction[halfWorkgroupSize];

  vec3 removeZero(in vec3 v) {
    if (v.x == 0.f) v.x = 1.f;
    if (v.y == 0.f) v.y = 1.f;
    if (v.z == 0.f) v.z = 1.f;
    return v;
  }

  void main() {
    uint lid = gl_LocalInvocationIndex.x;

    // Compute local bounds for this unit
    vec3 min_local = vec3(1e38);
    vec3 max_local = vec3(-1e38);
    for (uint i = lid; i < num_points; i += workgroupSize) {
      vec3 pos = positions[i];
      min_local = min(pos, min_local);
      max_local = max(pos, max_local);
    }

    // Perform reduce-add over all points
    if (lid >= halfWorkgroupSize) {
      min_reduction[lid - halfWorkgroupSize] = min_local;
      max_reduction[lid - halfWorkgroupSize] = max_local;
    }
    barrier();
    if (lid < halfWorkgroupSize) {
      min_reduction[lid] = min(min_local, min_reduction[lid]);
      max_reduction[lid] = max(max_local, max_reduction[lid]);
    }
    for (uint i = halfWorkgroupSize / 2; i > 1; i /= 2) {
      barrier();
      if (lid < i) {
        min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + i]);
        max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + i]);
      }
    }
    barrier();
    // perform store in starting unit
    if (lid == 0) {
      min_local = min(min_reduction[0], min_reduction[1]);
      max_local = max(max_reduction[0], max_reduction[1]);
      vec3 padding = (max_local - min_local) * 0.5f * padding;
      minBounds = min_local - padding;
      maxBounds = max_local + padding;
      range = (maxBounds - minBounds);
      invRange = 1.f / removeZero(range);
    }
  }
);

// Copied from compute_shader.glsl, adapted for 3d
GLSL(centering_src, 430,
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform bool scale;
  uniform float diameter;

  void main() {
    const uint workGroupID = gl_WorkGroupID.x;
    const uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points) {
      return;
    }

    vec3 center = (minBounds + maxBounds) * 0.5;
    vec3 pos = Positions[i];

    if (scale) {
      if (range.y < diameter) {
        float scale_factor = diameter * invRange.y;
        pos -= center;
        pos *= scale_factor;
      }
    } else {
      pos -= center;
    }

    Positions[i] = pos;
  }
);