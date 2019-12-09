#ifndef __APPLE__

#pragma once

#include <array>
#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "field_depth_computation.h"

namespace hdi::dr {
  class GpgpuDepthSneCompute {
  public:
    // Internal types
    typedef float scalar_t;
    typedef hdi::data::Embedding<scalar_t> embedding_t;
    typedef std::vector<hdi::data::MapMemEff<uint32_t, scalar_t>> sparse_matrix_t;

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

    GpgpuDepthSneCompute();
    ~GpgpuDepthSneCompute();

    // Initialize gpu components for computation
    void initialize();

    // Remove gpu components
    void clean();

    // Perform computation for specified iteration
    void compute();

    Bounds3D bounds() const {
      return _bounds;
    }

  private:
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
    DepthFieldComputation _depthFieldComputation;
    TsneParameters _params;
    Bounds3D _bounds;
  };
}

#endif