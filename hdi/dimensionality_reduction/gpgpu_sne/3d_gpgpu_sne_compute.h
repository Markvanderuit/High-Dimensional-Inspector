#ifndef __APPLE__

#pragma once

#include <array>
#include <cstdlib>
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "field_computation.h"
#include "3d_utils.h"

namespace hdi::dr {
  class Gpgpu3dSneCompute {
  public:
    // Base constr.
    Gpgpu3dSneCompute();
    ~Gpgpu3dSneCompute();

    // Initialize gpu components for computation
    void initialize(const embedding_t* embedding, 
                    const TsneParameters& params, 
                    const sparse_matrix_t& P);

    // Remove gpu components
    void clean();

    // Perform computation for specified iteration
    void compute(embedding_t* embedding, 
                 float exaggeration, 
                 float iteration, 
                 float mult);
    
    Bounds3D bounds() const {
      return _bounds;
    }
    
  private:
    void computeBounds(unsigned n, float padding);
    void interpolateFields(unsigned n, float* sum_Q);
    void sumQ(unsigned in, float* sum_Q);
    void computeGradients(unsigned n, float sum_Q, float exaggeration);
    void updatePoints(unsigned n, float iteration, float mult);
    void updateEmbedding(unsigned n, float exaggeration, float iteration);

  private:
    bool _initialized;
    bool _adaptive_resolution;
    float _resolution_scaling;

    std::array<GLuint, 10> _buffers;
    std::array<ShaderProgram, 6> _programs;
    Compute3DFieldComputation _fieldComputation;
    TsneParameters _params;
    Bounds3D _bounds;
  };
}

#endif