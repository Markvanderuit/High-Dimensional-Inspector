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

GLSL(expansion_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer Neig { uint neighboursBuffer[]; };
  layout(binding = 1, std430) restrict coherent buffer Layu { uint sizesBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uint kNeighbours;

  bool containsk(uint j, uint i) {
    // Unfortunately the list is unsorted by indices, so search is linear
    for (uint k = 1; k < kNeighbours; k++) {
      if (neighboursBuffer[j * kNeighbours + k] == i) {
        return true;
      }
    }
    return false;
  }

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPoints) {
      return;
    }

    atomicAdd(sizesBuffer[i], kNeighbours - 1); // remove 1 to remove self from own indices

    for (uint k = 1; k < kNeighbours; k++) {
      // Get index of neighbour
      const uint j = neighboursBuffer[i * kNeighbours + k];
      
      // Find index of self in neighbour's list of neighbours
      // Self not found, add 1 to neighbour's size
      if (!containsk(j, i)) {
        atomicAdd(sizesBuffer[j], 1);
      }
    }
  }
);

GLSL(layout_src, 450,
  struct Layout {
    uint offset;
    uint size;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z  = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer Scan { uint scanBuffer[]; };
  layout(binding = 1, std430) restrict writeonly buffer Lay { Layout layoutBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPoints) {
      return;
    }

    // Update layout based on results of prefix sum
    Layout l;
    if (i == 0) {
      l = Layout(0, scanBuffer[0]);
    } else {
      l = Layout(scanBuffer[i - 1], scanBuffer[i] - scanBuffer[i - 1]);
    }
    layoutBuffer[i] = l;
  }
);

GLSL(neighbours_src, 450,
  struct Layout {
    uint offset;
    uint size;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z  = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer ANei { uint asymmetricNeighboursBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer ASym { float asymmetricSimilaritiesBuffer[]; };
  layout(binding = 2, std430) restrict coherent buffer Coun { uint countBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer Layu { Layout layoutBuffer[]; };
  layout(binding = 4, std430) restrict writeonly buffer Nei { uint symmetricNeighboursBuffer[]; };
  layout(binding = 5, std430) restrict writeonly buffer Sym { float symmetricSimilaritiesBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uint kNeighbours;

  uint searchk(uint j, uint i) {
    // Unfortunately the list is unsorted, so search is linear
    for (uint k = 1; k < kNeighbours; k++) {
      if (asymmetricNeighboursBuffer[j * kNeighbours + k] == i) {
        return k;
      }
    }
    return 0;
  }

  void main() {
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / kNeighbours;
    const uint k = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) % kNeighbours;

    if (i >= nPoints) {
      return;
    }

    // TODO Optimize this so the first index is just intrinsically not part of the computation
    if (k == 0) {
      return;
    }

    // Read original asymmetric neighbours
    const uint j = asymmetricNeighboursBuffer[i * kNeighbours + k];
    const float f = asymmetricSimilaritiesBuffer[i * kNeighbours + k];

    // Read offsets and sizes
    const Layout li = layoutBuffer[i];
    const Layout lj = layoutBuffer[j];
    
    // Copy over to own neighbour indices
    symmetricNeighboursBuffer[li.offset + k - 1] = j;

    // Potentially copy self over to neighbour's neighbour indices, if they don't already know it
    uint _k = searchk(j, i);
    if (_k == 0) {
      // Copy self to neighbour's indices, as they don't have ours yet
      _k = atomicAdd(countBuffer[j], 1);
      symmetricNeighboursBuffer[lj.offset + kNeighbours - 1 + _k] = i;

      // Symmetrize similarities
      const float v = 0.5 * f;
      symmetricSimilaritiesBuffer[li.offset + k - 1] = v;
      symmetricSimilaritiesBuffer[lj.offset + (kNeighbours - 1) + _k] = v;
    } else if (j > i) {
      // Symmetrize similarities
      const float v = 0.5 * (f + asymmetricSimilaritiesBuffer[j * kNeighbours + _k]);
      symmetricSimilaritiesBuffer[li.offset + k - 1] = v;
      symmetricSimilaritiesBuffer[lj.offset + _k - 1] = v;
    }
  }
);

GLSL(similarity_src, 450,
  /*
   * WARNING: unreadable code ahead, bluntly ported to GLSL. Originals are from
   * hd_joint_probability_generator.h/cpp/inl and math_utils.h/cpp/inl.
   */

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer Indi { uint neighboursBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Dist { float distancesBuffer[]; };
  layout(binding = 2, std430) restrict buffer SimilaritiesM { float similaritiesBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uint kNeighbours;
  layout(location = 2) uniform float perplexity;
  layout(location = 3) uniform uint nIters;
  layout(location = 4) uniform float epsilon;

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPoints) {
      return;
    }

    // Use beta instead of sigma, as per BH-SNE code
    float beta = 1.f;
    float lowerBoundBeta = -FLT_MAX;
    float upperBoundBeta = FLT_MAX;
    float sum = FLT_MIN;
    
    // Perform a binary search for the best fit for Beta
    bool foundBeta = false;
    uint iter = 0;
    while (!foundBeta && iter < nIters) {
      // Compute P_ij with current value of Beta (Sigma, actually)
      // Ignore j = 0, as that is i itself in the local neighbourhood
      sum = FLT_MIN;
      for (uint j = 1; j < kNeighbours; j++) {
        const uint ij = i * kNeighbours + j;
        const float v = exp(-beta * distancesBuffer[ij]);
        similaritiesBuffer[ij] = v;
        sum += v;
      }

      // Compute entropy over the current gaussian's values
      // float negativeEntropy = 0.f;
      float entropy = 0.f;
      for (uint j = 1; j < kNeighbours; j++) {
        const uint ij = i * kNeighbours + j;
        const float v = similaritiesBuffer[ij];
        const float e = v * log(v);
        // negativeEntropy += (e != e) ? 0 : e;
        entropy += beta * distancesBuffer[ij] * similaritiesBuffer[ij];
      }
      // negativeEntropy *= -1.f;
      entropy = (entropy / sum) + log(sum);

      // Test if the difference falls below epsilon
      // float entropy = (negativeEntropy / sum) + log(sum);
      float entropyDiff = entropy - log(perplexity);
      foundBeta = entropyDiff < epsilon && -entropyDiff < epsilon;

      // Tighten bounds for binary search
      if (!foundBeta) {
        if (entropyDiff > 0) {
          lowerBoundBeta = beta;
          beta = (upperBoundBeta == FLT_MAX || upperBoundBeta == -FLT_MAX)
               ? 2.f * beta
               : 0.5f * (beta + upperBoundBeta);
        } else {
          upperBoundBeta = beta;
          beta = (lowerBoundBeta == FLT_MAX || lowerBoundBeta == -FLT_MAX)
               ? 0.5f * beta
               : 0.5f * (beta + lowerBoundBeta);
        }
      }
      
      iter++;
    }

    // Normalize kernel at the end
    if (!foundBeta) {
      const float v = 1.f / float(kNeighbours - 1);
      for (uint j = 1; j < kNeighbours; j++) {
        const uint ij = i * kNeighbours + j;
        similaritiesBuffer[ij] = v;
      }
    } else {
      const float div = 1.f / sum;
      for (uint j = 1; j < kNeighbours; j++) {
        const uint ij = i * kNeighbours + j;
        similaritiesBuffer[ij] *= div;
      }
    }
  }
);

GLSL(symmetrize_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer Indi { uint neighboursBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Inte { float intermediateBuffer[]; };
  layout(binding = 2, std430) restrict buffer SimilaritiesM { float similaritiesBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uint kNeighbours;

  uint searchk(uint j, uint i) {
    // Unfortunately the list is unsorted, so search is linear
    for (uint k = 1; k < kNeighbours; k++) {
      if (neighboursBuffer[j * kNeighbours + k] == i) {
        return k;
      }
    }
    return 0;
  }

  void main() {
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / kNeighbours;
    const uint k = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) % kNeighbours;
    const uint ij = i * kNeighbours + k;

    const uint j = neighboursBuffer[ij];
    const uint _k = searchk(j, i);
    const uint ji = j * kNeighbours + _k;

    if (_k == 0) {
      // No other value for p_j|i found, bad luck with kNN
      similaritiesBuffer[ij] = intermediateBuffer[ij];
    } else {
      // Apply p_ij = (p_i|j + p_j|i) / 2
      const float v = 0.5f * (intermediateBuffer[ij] + intermediateBuffer[ji]);
      similaritiesBuffer[ij] = v;
    }
  }
);

GLSL(symmetrize_2_src, 450,
  struct Layout {
    uint offset;
    uint size;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_1  = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer Indi { uint neighboursBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer Inte { float intermediateBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer Coun { uint countBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer Layo { Layout layoutBuffer[]; };
  layout(binding = 4, std430) restrict buffer SimilaritiesM { float similaritiesBuffer[]; };

  // Uniform values
  layout(location = 0) uniform uint nPoints;
  layout(location = 1) uniform uint kNeighbours;

  uint searchk(uint j, uint i) {
    // Unfortunately the list is unsorted, so search is linear
    for (uint k = 1; k < kNeighbours; k++) {
      if (neighboursBuffer[j * kNeighbours + k] == i) {
        return k;
      }
    }
    return 0;
  }

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= nPoints) {
      return;
    }

    // Grab layout
    Layout l = layoutBuffer[i];

    // Go over neighbours
    for (uint k = 0; k < kNeighbours - 1; k++) {
      const uint j = neighboursBuffer[ * kNeighbours + k];
      uint _k = searchk(j, i);
      float v = 0.f;
      if (_k == 0 && i != 0) {
        // Other does not store this point as a neighbor
        _k = kNeighbours + atomicAdd(countBuffer[j]);
        v = 0.5 * intermediateBuffer[i * kNeighbours + j];
      } else if (j > i) {
        // Other stores this point as a neighbor
        v = 0.5 * (intermediateBuffer[i * kNeighbours + j] + intermediateBuffer[j * kNeighbours + i]);
      }
      
      const uint ij = layoutBuffer[i].offset + k
      const uint ji = layoutBuffer[j].offset + _k;
      similaritiesBuffer[ij] = v;
      similaritiesBuffer[ji] = v;
    }  
  }
);