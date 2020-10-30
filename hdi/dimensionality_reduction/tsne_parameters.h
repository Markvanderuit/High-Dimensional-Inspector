/*
*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
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
*
*/

#pragma once
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"

namespace hdi::dr {
  class TsneParameters {
  private:
    using uint = unsigned;

  public:
    // Input/output data parameters
    uint n = 0;
    uint nHighDimensions = 0;
    uint nLowDimensions = 2;

    // Base tSNE parameters
    uint iterations = 1000;
    float perplexity = 30.f;

    // RNG initialization parameters
    int seed = 1;
    float rngRange = 0.1f;

    // Barnes-Hut approximation parameters
    float singleHierarchyTheta = 0.5f;
    float dualHierarchyTheta = 0.5f;

    // Gradient descent switching iterations
    uint momentumSwitchIter = 250;
    uint removeExaggerationIter = 250;
    uint exponentialDecayIter = 150;

    // Gradient descent parameters
    float minimumGain = 0.1f;
    float eta = 200.f;
    float momentum = GRAD_MOMENTUM;
    float finalMomentum = GRAD_FINAL_MOMENTUM;
    float exaggerationFactor = GRAD_EXAGGERATION_FACTOR;
  };
} // hdi::dr
