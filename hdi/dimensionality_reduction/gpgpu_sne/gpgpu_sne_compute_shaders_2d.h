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

#define SHADER_SRC(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

namespace hdi::dr::_2d {
  SHADER_SRC(bounds_src, 450,
    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Pos { vec2 positions[]; };
    layout(binding = 1, std430) restrict buffer BoundsReduce {
      vec2 minBoundsReduceAdd[128];
      vec2 maxBoundsReduceAdd[128];
    };
    layout(binding = 2, std430) restrict writeonly buffer Bounds { 
      vec2 minBounds;
      vec2 maxBounds;
      vec2 range;
      vec2 invRange;
    };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform uint iter;
    layout(location = 2) uniform float padding;

    const uint halfGroupSize = gl_WorkGroupSize.x / 2;
    shared vec2 min_reduction[halfGroupSize];
    shared vec2 max_reduction[halfGroupSize];

    void main() {
      const uint lid = gl_LocalInvocationIndex.x;
      vec2 min_local = vec2(1e38);
      vec2 max_local = vec2(-1e38);

      if (iter == 0) {
        // First iteration adds all values
        for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
            i < nPoints;
            i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
          vec2 pos = positions[i];
          min_local = min(pos, min_local);
          max_local = max(pos, max_local);
        }
      } else if (iter == 1) {
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
        if (iter == 0) {
          minBoundsReduceAdd[gl_WorkGroupID.x] = min_local;
          maxBoundsReduceAdd[gl_WorkGroupID.x] = max_local;
        } else if (iter == 1) {
          vec2 padding = (max_local - min_local) * 0.5f * padding;
          vec2 _minBounds = min_local - padding;
          vec2 _maxBounds = max_local + padding;
          vec2 _range = (_maxBounds - _minBounds);

          maxBounds = _maxBounds;
          minBounds = _minBounds;
          range = _range;
          // Prevent div-by-0, set 0 to 1
          invRange = 1.f / (_range + vec2(equal(_range, vec2(0))));
        }
      }
    }
  );

  SHADER_SRC(sumq_src, 450,
    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Value { vec3 Values[]; };
    layout(binding = 1, std430) restrict buffer SumReduce { float SumReduceAdd[128]; };
    layout(binding = 2, std430) restrict writeonly buffer Sum { 
      float sumQ;
      float invSumQ;
    };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform uint iter;
    
    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared float reduction_array[halfGroupSize];

    void main() {
      uint lid = gl_LocalInvocationID.x;
      float sum = 0.f;
      if (iter == 0) {
        // First iteration adds all values
        for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
            i < nPoints;
            i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
          sum += max(Values[i].x - 1.f, 0f);
        }
      } else if (iter == 1) {
        // Second iteration adds resulting 128 values
        sum = SumReduceAdd[lid];      
      }

      // Reduce add to a single value
      if (lid >= halfGroupSize) {
        reduction_array[lid - halfGroupSize] = sum;
      }
      barrier();
      if (lid < halfGroupSize) {
        reduction_array[lid] += sum;
      }
      for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
        barrier();
        if (lid < i) {
          reduction_array[lid] += reduction_array[lid + i];
        }
      }
      barrier();
      if (lid < 1) {
        if (iter == 0) {
          SumReduceAdd[gl_WorkGroupID.x] = reduction_array[0] + reduction_array[1];
        } else if (iter == 1) {
          float _sumQ = reduction_array[0] + reduction_array[1];
          sumQ = _sumQ;
          invSumQ = 1.f / _sumQ;
        }
      }
    }
  );

  SHADER_SRC(positive_forces_src, 450,
    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Pos { vec2 Positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Neigh { uint Neighbours[]; };
    layout(binding = 2, std430) restrict readonly buffer Prob { float Probabilities[]; };
    layout(binding = 3, std430) restrict readonly buffer Ind { int Indices[]; };
    layout(binding = 4, std430) restrict writeonly buffer Posit { vec2 PositiveForces[]; };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform float invNPoints;

    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec2 reduction_array[halfGroupSize];

    void main () {
      const uint i = gl_WorkGroupID.x;
      const uint lid = gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      // Computing positive forces using nearest neighbors
      vec2 point_i = Positions[i];
      int index = Indices[i * 2 + 0];
      int size = Indices[i * 2 + 1];
      vec2 positive_force = vec2(0);
      for (uint j = lid; j < size; j += groupSize) {
        // Get other point coordinates
        vec2 point_j = Positions[Neighbours[index + j]];

        // Calculate 2D distance between the two points
        vec2 dist = point_i - point_j;

        // Similarity measure of the two points
        float qij = 1.f + dot(dist, dist);

        // Calculate the attractive force
        positive_force += Probabilities[index + j] * dist / qij;
      }
      positive_force *= invNPoints;

      // Reduce add positive_force to a single value
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

  SHADER_SRC(gradients_src, 450,
    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Pos { vec2 PositiveForces[]; };
    layout(binding = 1, std430) restrict readonly buffer Field { vec3 Fields[]; };
    layout(binding = 2, std430) restrict readonly buffer SumQ { 
      float sumQ;
      float invSumQ;
    };
    layout(binding = 3, std430) restrict writeonly buffer Grad { vec2 Gradients[]; };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform float exaggeration;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      vec2 positive = PositiveForces[i];
      vec2 negative = Fields[i].yz * invSumQ;
      Gradients[i] = 4.f * (exaggeration * positive - negative);
    }
  );

  SHADER_SRC(update_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict buffer Pos { vec2 Positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Grad { vec2 Gradients[]; };
    layout(binding = 2, std430) restrict buffer PrevGrad { vec2 PrevGradients[]; };
    layout(binding = 3, std430) restrict buffer Gains { vec2 Gain[]; };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform float eta;
    layout(location = 2) uniform float minGain;
    layout(location = 3) uniform float mult;
    layout(location = 4) uniform float iterMult;

    void main() {
      // Invocation location
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
      if (i >= nPoints) {
        return;
      }
      
      // Load values
      vec2 grad = Gradients[i];
      vec2 pgrad = PrevGradients[i];
      vec2 gain = Gain[i];

      // Compute gain, clamp at minGain
      vec2 gainDir = vec2(!equal(sign(grad), sign(pgrad)));
      gain = mix(gain * 0.8f, 
        gain + 0.2f, 
        gainDir);
      gain = max(gain, minGain);

      // Compute gradient
      vec2 etaGain = eta * gain;
      grad = fma(
        vec2(greaterThan(grad, vec2(0))), 
        vec2(2f), 
        vec2(-1f)
      ) * abs(grad * etaGain) / etaGain;

      // Compute previous gradient
      pgrad = fma(
        pgrad,
        vec2(iterMult),
        -etaGain * grad
      );

      // Store values again
      Gain[i] = gain;
      PrevGradients[i] = pgrad;
      Positions[i] += pgrad * mult;
    }
  );

  SHADER_SRC(center_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict buffer Pos { vec2 Positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Bounds { 
      vec2 minBounds;
      vec2 maxBounds;
      vec2 range;
      vec2 invRange;
    };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform vec2 center;
    layout(location = 2) uniform float scaling;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
      if (i >= nPoints) {
        return;
      }
      
      Positions[i] = scaling * (Positions[i] - center); 
    }
  );
}