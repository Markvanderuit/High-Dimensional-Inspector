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

namespace hdi::dr {
  namespace _2d {
    SHADER_SRC(qij_src, 450,
      layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
      layout(binding = 0, std430) restrict readonly buffer Pos { vec2 posBuffer[]; };
      layout(binding = 1, std430) restrict writeonly buffer Qij { float qijBuffer[]; };
      layout(location = 0) uniform uint nPoints;
      layout(location = 1) uniform uint begin;

      const uint groupSize = gl_WorkGroupSize.x;
      const uint halfGroupSize = groupSize / 2;
      shared float reductionArray[halfGroupSize];

      void main() {
        uint gid = begin + gl_WorkGroupID.x;
        uint lid = gl_LocalInvocationIndex.x;

        vec2 pos = posBuffer[gl_WorkGroupID.x];

        // Iterate over points to obtain density/gradient
        float q_ij = 0.0;
        for (uint i = lid; i < nPoints; i += groupSize) {
          vec2 t = pos - posBuffer[i];
          q_ij += 1.0 / (1.0 + dot(t, t));
        }
        
        // Perform reduce add over all computed values for this point
        if (lid >= halfGroupSize) {
          reductionArray[lid - halfGroupSize] = q_ij;
        }
        barrier();
        if (lid < halfGroupSize) {
          reductionArray[lid] += q_ij;
        }
        for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
          barrier();
          if (lid < i) {
            reductionArray[lid] += reductionArray[lid + i];
          }
        }
        barrier();
        if (lid < 1) {
          qijBuffer[gid] = reductionArray[0] + reductionArray[1];
        }
      }
    );

    SHADER_SRC(reduce_src, 450,
      layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
      layout(binding = 0, std430) restrict readonly buffer Input { float inputBuffer[]; };
      layout(binding = 1, std430) restrict buffer Intermediate { float intermediateBuffer[256]; };
      layout(binding = 2, std430) restrict writeonly buffer Final { float finalBuffer; };
      layout(location = 0) uniform uint nPoints;
      layout(location = 1) uniform uint iter;
      
      const uint groupSize = gl_WorkGroupSize.x;
      const uint halfGroupSize = groupSize / 2;
      shared float reductionArray[halfGroupSize];

      void main() {
        uint lid = gl_LocalInvocationID.x;
        float sum = 0.f;
        if (iter == 0) {
          // First iteration reduces n to 128
          for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
              i < nPoints;
              i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
            sum += inputBuffer[i];
          }
        } else if (iter == 1) {
          // Second iteration reduces 128 to 1
          sum = intermediateBuffer[lid];      
        }

        // Reduce add to a single value
        if (lid >= halfGroupSize) {
          reductionArray[lid - halfGroupSize] = sum;
        }
        barrier();
        if (lid < halfGroupSize) {
          reductionArray[lid] += sum;
        }
        for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
          barrier();
          if (lid < i) {
            reductionArray[lid] += reductionArray[lid + i];
          }
        }
        barrier();
        if (lid < 1) {
          if (iter == 0) {
            intermediateBuffer[gl_WorkGroupID.x] = reductionArray[0] + reductionArray[1];
          } else if (iter == 1) {
            finalBuffer = reductionArray[0] + reductionArray[1];
          }
        }
      }
    );

    SHADER_SRC(kl_src, 450,
      layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
      layout(binding = 0, std430) restrict readonly buffer Posi { vec2 posBuffer[]; };
      layout(binding = 1, std430) restrict readonly buffer SumQ { float sumQBuffer; };
      layout(binding = 2, std430) restrict readonly buffer Neig { uint neighbsBuffer[]; };
      layout(binding = 3, std430) restrict readonly buffer Prob { float probsBuffer[]; };
      layout(binding = 4, std430) restrict readonly buffer Indi { int indsBuffer[]; };
      layout(binding = 5, std430) restrict writeonly buffer KLD { float klBuffer[]; };
      layout(location = 0) uniform uint nPoints;
      layout(location = 1) uniform uint begin;

      const uint groupSize = gl_WorkGroupSize.x;
      const uint halfGroupSize = groupSize / 2;
      shared float reductionArray[halfGroupSize];

      void main() {
        const uint gid = begin + gl_WorkGroupID.x;
        const uint lid = gl_LocalInvocationID.x;

        // Load buffer data
        float invSumQ = 1.0 / sumQBuffer;
        vec2 pos = posBuffer[gid];
        int idx = indsBuffer[gid * 2 + 0];
        int nIdx = indsBuffer[gid * 2 + 1];

        // Sum over nearest neighbours data with a full workgroup
        float klc = 0.0;
        for (uint i = lid; i < nIdx; i += groupSize) {
          float p_ij = probsBuffer[idx + i] / (2.0 * float(nPoints)); // TODO why over 2N?
          if (p_ij != 0.f) {
            vec2 t = pos - posBuffer[neighbsBuffer[idx + i]];
            float q_ij = 1.0 / (1.0 + dot(t, t));
            float v = p_ij / (q_ij * invSumQ);
            klc += p_ij * log(v);
          }
        }

        // Reduce add to a single value
        if (lid >= halfGroupSize) {
          reductionArray[lid - halfGroupSize] = klc;
        }
        barrier();
        if (lid < halfGroupSize) {
          reductionArray[lid] += klc;
        }
        for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
          barrier();
          if (lid < i) {
            reductionArray[lid] += reductionArray[lid + i];
          }
        }
        barrier();
        if (lid < 1) {
          klBuffer[gid] = reductionArray[0] + reductionArray[1];
        }
      }
    );
  } // namespace _2d

  namespace _3d {
    SHADER_SRC(qij_src, 450,
      layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
      layout(binding = 0, std430) restrict readonly buffer Pos { vec3 posBuffer[]; };
      layout(binding = 1, std430) restrict writeonly buffer Qij { float qijBuffer[]; };
      layout(location = 0) uniform uint nPoints;
      layout(location = 1) uniform uint begin;

      const uint groupSize = gl_WorkGroupSize.x;
      const uint halfGroupSize = groupSize / 2;
      shared float reductionArray[halfGroupSize];

      void main() {
        uint gid = begin + gl_WorkGroupID.x;
        uint lid = gl_LocalInvocationIndex.x;

        vec3 pos = posBuffer[gl_WorkGroupID.x];

        // Iterate over points to obtain density/gradient
        float q_ij = 0.0;
        for (uint i = lid; i < nPoints; i += groupSize) {
          vec3 t = pos - posBuffer[i];
          q_ij += 1.0 / (1.0 + dot(t, t));
        }
        
        // Perform reduce add over all computed values for this point
        if (lid >= halfGroupSize) {
          reductionArray[lid - halfGroupSize] = q_ij;
        }
        barrier();
        if (lid < halfGroupSize) {
          reductionArray[lid] += q_ij;
        }
        for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
          barrier();
          if (lid < i) {
            reductionArray[lid] += reductionArray[lid + i];
          }
        }
        barrier();
        if (lid < 1) {
          qijBuffer[gid] = reductionArray[0] + reductionArray[1];
        }
      }
    );

    SHADER_SRC(kl_src, 450,
      layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
      layout(binding = 0, std430) restrict readonly buffer Posi { vec3 posBuffer[]; };
      layout(binding = 1, std430) restrict readonly buffer SumQ { float sumQBuffer; };
      layout(binding = 2, std430) restrict readonly buffer Neig { uint neighbsBuffer[]; };
      layout(binding = 3, std430) restrict readonly buffer Prob { float probsBuffer[]; };
      layout(binding = 4, std430) restrict readonly buffer Indi { int indsBuffer[]; };
      layout(binding = 5, std430) restrict writeonly buffer KLD { float klBuffer[]; };
      layout(location = 0) uniform uint nPoints;
      layout(location = 1) uniform uint begin;

      const uint groupSize = gl_WorkGroupSize.x;
      const uint halfGroupSize = groupSize / 2;
      shared float reductionArray[halfGroupSize];

      void main() {
        const uint gid = begin + gl_WorkGroupID.x;
        const uint lid = gl_LocalInvocationID.x;

        // Load buffer data
        float invSumQ = 1.0 / sumQBuffer;
        vec3 pos = posBuffer[gid];
        int idx = indsBuffer[gid * 2 + 0];
        int nIdx = indsBuffer[gid * 2 + 1];

        // Sum over nearest neighbours data with a full workgroup
        float klc = 0.0;
        for (uint i = lid; i < nIdx; i += groupSize) {
          float p_ij = probsBuffer[idx + i] / (2.0 * float(nPoints)); // TODO why over 2N?
          if (p_ij != 0.f) {
            vec3 t = pos - posBuffer[neighbsBuffer[idx + i]];
            float q_ij = 1.0 / (1.0 + dot(t, t));
            float v = p_ij / (q_ij * invSumQ);
            klc += p_ij * log(v);
          }
        }

        // Reduce add to a single value
        if (lid >= halfGroupSize) {
          reductionArray[lid - halfGroupSize] = klc;
        }
        barrier();
        if (lid < halfGroupSize) {
          reductionArray[lid] += klc;
        }
        for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
          barrier();
          if (lid < i) {
            reductionArray[lid] += reductionArray[lid + i];
          }
        }
        barrier();
        if (lid < 1) {
          klBuffer[gid] = reductionArray[0] + reductionArray[1];
        }
      }
    );
  } // namespace _3d
} // namespace hdi::dr