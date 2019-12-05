#ifndef __APPLE__

#pragma once

#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "field_computation.h"
#include <array>
#include <cstdlib>

namespace hdi::dr {
  class Gpgpu3dSneCompute {
  public:
    typedef hdi::data::Embedding<float> embedding_t;
    typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_matrix_t;

    // Simple bounds representation
    // vec3 gets padded to vec4 in std430, so we align to 16 bytes
    struct alignas(16) Point3D { float x, y, z; };
    struct Bounds3D {
      Point3D min, max;
      
      Point3D getRange() const {
        return Point3D { 
          max.x - min.x, 
          max.y - min.y, 
          max.z - min.z 
        };
      }
    };

    // Linearized representation of sparse probability matrix
    struct LinearProbabilityMatrix {
      std::vector<uint32_t> neighbours;
      std::vector<float> probabilities;
      std::vector<int> indices;
    };

    // Base constr.
    Gpgpu3dSneCompute();

    void initialize(const embedding_t* embedding, 
                    const TsneParameters& params, 
                    const sparse_matrix_t& P);
    void clean();
    void compute(embedding_t* embedding, 
                 float exaggeration, 
                 float iteration, 
                 float mult);
    
  private:
    void computeBounds(unsigned n, float padding);
    void interpolateFields(unsigned n, float* sum_Q);
    void sumQ(unsigned in, float* sum_Q);
    void computeGradients(unsigned n, float sum_Q, float exaggeration);
    void updatePoints(unsigned n, float iteration, float mult);
    void updateEmbedding(unsigned n, float exaggeration, float iteration);

    bool _initialized;
    bool _adaptive_resolution;
    float _resolution_scaling;

    ShaderProgram _interpolation_program;
    ShaderProgram _sumq_program;
    ShaderProgram _forces_program;
    ShaderProgram _update_program;
    ShaderProgram _bounds_program;
    ShaderProgram _centering_program;

    std::array<GLuint, 10> _buffers;
    Compute3DFieldComputation _fieldComputation;
    TsneParameters _params;
    Bounds3D _bounds;
  };
}

#endif