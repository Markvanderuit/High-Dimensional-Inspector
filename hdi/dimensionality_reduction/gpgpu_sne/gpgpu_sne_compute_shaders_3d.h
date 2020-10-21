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

#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

namespace hdi::dr::_3d {
  GLSL(bounds_src, 450,
    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Pos { vec3 positions[]; };
    layout(binding = 1, std430) restrict buffer BoundsReduce {
      vec3 minBoundsReduceAdd[128];
      vec3 maxBoundsReduceAdd[128];
    };
    layout(binding = 2, std430) restrict writeonly buffer Bounds { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform uint iter;
    layout(location = 2) uniform float padding;

    const uint halfGroupSize = gl_WorkGroupSize.x / 2;
    shared vec3 min_reduction[halfGroupSize];
    shared vec3 max_reduction[halfGroupSize];

    void main() {
      const uint lid = gl_LocalInvocationIndex.x;
      vec3 min_local = vec3(1e38);
      vec3 max_local = vec3(-1e38);

      if (iter == 0) {
        // First iteration adds all values
        for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
            i < nPoints;
            i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
          vec3 pos = positions[i];
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
          vec3 padding = (max_local - min_local) * 0.5f * padding;
          vec3 _minBounds = min_local - padding;
          vec3 _maxBounds = max_local + padding;
          vec3 _range = (_maxBounds - _minBounds);

          maxBounds = _maxBounds;
          minBounds = _minBounds;
          range = _range;
          // Prevent div-by-0, set 0 to 1
          invRange = 1.f / (_range + vec3(equal(_range, vec3(0))));
        }
      }
    }
  );

  GLSL(sumq_src, 450,
    layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict readonly buffer Value { vec4 Values[]; };
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

  GLSL(positive_forces_src, 450,
    GLSL_PROTECT( #extension GL_KHR_shader_subgroup_ballot : require )
    GLSL_PROTECT( #extension GL_KHR_shader_subgroup_arithmetic : require )

    struct Layout {
      uint offset;
      uint size;
    };

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    // Buffer bindings
    layout(binding = 0, std430) restrict readonly buffer Posi { vec3 positionsBuffer[]; };
    layout(binding = 1, std430) restrict readonly buffer Layo { Layout layoutsBuffer[]; };
    layout(binding = 2, std430) restrict readonly buffer Neig { uint neighboursBuffer[]; };
    layout(binding = 3, std430) restrict readonly buffer Simi { float similaritiesBuffer[]; };
    layout(binding = 4, std430) restrict writeonly buffer Att { vec3 attrForcesBuffer[]; };

    // Uniform values
    layout(location = 0) uniform uint nPos;
    layout(location = 1) uniform float invPos;

    // Shorthand subgroup constants
    const uint thread = gl_SubgroupInvocationID;
    const uint nThreads = gl_SubgroupSize;

    void main() {
      const uint i = (gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x) / nThreads;
      if (i >= nPos) {
        return;
      }

      // Load data for subgroup
      const vec3 position = subgroupBroadcastFirst(thread < 1 ? positionsBuffer[i] : vec3(0));

      // Sum attractive force over nearest neighbours
      Layout l = layoutsBuffer[i];
      vec3 attrForce = vec3(0);
      for (uint ij = l.offset + thread; ij < l.offset + l.size; ij += nThreads) {
        // Calculate difference between the two positions
        const vec3 diff = position - positionsBuffer[neighboursBuffer[ij]];

        // High/low dimensional similarity measures of i and j
        const float p_ij = similaritiesBuffer[ij];
        const float q_ij = 1.f / (1.f + dot(diff, diff));

        // Calculate the attractive force
        attrForce += p_ij * q_ij * diff;
      }
      attrForce = subgroupAdd(attrForce * invPos);

      // Store result
      if (thread < 1) {
        attrForcesBuffer[i] = attrForce;
      }
    }
  );

  GLSL(gradients_src, 450,
    // Wrapper structure for sum of Q data
    struct SumQ {
      float z;
      float inv;
    };
    
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    layout(binding = 0, std430) restrict readonly buffer Attr { vec3 attrForcesBuffer[]; };
    layout(binding = 1, std430) restrict readonly buffer Fiel { vec4 fieldBuffer[]; };
    layout(binding = 2, std430) restrict readonly buffer SumB { SumQ sumq; };
    layout(binding = 3, std430) restrict writeonly buffer Gra { vec3 gradientsBuffer[]; };

    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform float exaggeration;

    void main() {
      const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= nPoints) {
        return;
      }

      // Load attractive and repulsive forces
      vec3 attrForce = attrForcesBuffer[i];
      vec3 repForce = fieldBuffer[i].yzw * sumq.inv;

      // Compute gradient
      gradientsBuffer[i] = 4.f * (exaggeration * attrForce - repForce);
    }
  );

  GLSL(update_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict buffer Pos { vec3 Positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Grad { vec3 Gradients[]; };
    layout(binding = 2, std430) restrict buffer PrevGrad { vec3 PrevGradients[]; };
    layout(binding = 3, std430) restrict buffer Gains { vec3 Gain[]; };
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
      vec3 grad = Gradients[i];
      vec3 pgrad = PrevGradients[i];
      vec3 gain = Gain[i];

      // Compute gain, clamp at minGain
      vec3 gainDir = vec3(!equal(sign(grad), sign(pgrad)));
      gain = mix(gain * 0.8f, 
        gain + 0.2f, 
        gainDir);
      gain = max(gain, minGain);

      // Compute gradient
      vec3 etaGain = eta * gain;
      grad = fma(
        vec3(greaterThan(grad, vec3(0))), 
        vec3(2f), 
        vec3(-1f)
      ) * abs(grad * etaGain) / etaGain;

      // Compute previous gradient
      pgrad = fma(
        pgrad,
        vec3(iterMult),
        -etaGain * grad
      );

      // Store values again
      Gain[i] = gain;
      PrevGradients[i] = pgrad;
      Positions[i] += pgrad * mult;
    }
  );

  GLSL(center_src, 450,
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0, std430) restrict buffer Pos { vec3 Positions[]; };
    layout(binding = 1, std430) restrict readonly buffer Bounds { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    layout(location = 0) uniform uint nPoints;
    layout(location = 1) uniform vec3 center;
    layout(location = 2) uniform float scaling;

    void main() {
      uint i = gl_WorkGroupID.x 
             * gl_WorkGroupSize.x
             + gl_LocalInvocationIndex.x;
      if (i >= nPoints) {
        return;
      }
      
      Positions[i] = scaling * (Positions[i] - center); 
    }
  );
}