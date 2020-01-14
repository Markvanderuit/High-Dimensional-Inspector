#ifndef __APPLE__

#pragma once

#define ASSERT_SUM_Q // Enable SUM_Q value checking
#define USE_DEPTH_FIELD // Enable depthmap leveraging field computation
// #define USE_MIPMAP // Enable mipmap leveraging field computation

#include <cstdlib>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "gpgpu_utils.h"
#include "3d_utils.h"

#ifdef USE_DEPTH_FIELD
#include "depth_field_computation.h"
#elif defined USE_MIPMAP
#include "mipmap_field_computation.h"
#else
#include "3d_field_computation.h"
#endif 

namespace hdi::dr {
  class Gpgpu3dSneCompute {
  public:
    // Base constr., destr.
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
                 unsigned iteration, 
                 float mult);
    
    Bounds3D bounds() const {
      return _bounds;
    }

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
      _fieldComputation.setLogger(logger);
    }
    
  private:
    void computeBounds(unsigned n, float padding);
    void sumQ(unsigned n);
    void computeGradients(unsigned n, float exaggeration);
    void updatePoints(unsigned n, float iteration, float mult);
    void updateEmbedding(unsigned n, float exaggeration, float iteration);

    bool _initialized;
    bool _adaptive_resolution;
    float _resolution_scaling;

    enum BufferType {
      // Enums matching to storage buffers in _buffers array
      BUFFER_POSITION,
      BUFFER_INTERP_FIELDS,
      BUFFER_SUM_Q,
      BUFFER_SUM_Q_REDUCE_ADD,
      BUFFER_NEIGHBOUR,
      BUFFER_PROBABILITIES,
      BUFFER_INDEX,
      BUFFER_POSITIVE_FORCES,
      BUFFER_GRADIENTS,
      BUFFER_PREV_GRADIENTS,
      BUFFER_GAIN,
      BUFFER_BOUNDS_REDUCE_ADD,
      BUFFER_BOUNDS,
      
      // Static enum length
      BufferTypeLength 
    };

    enum ProgramType {
      // Enums matching to shader programs in _programs array
      PROGRAM_SUM_Q,
      PROGRAM_POSITIVE_FORCES,
      PROGRAM_GRADIENTS,
      PROGRAM_UPDATE,
      PROGRAM_BOUNDS,
      PROGRAM_CENTERING,
      
      // Static enum length
      ProgramTypeLength 
    };

    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
#ifdef USE_DEPTH_FIELD
    DepthFieldComputation _fieldComputation;
#elif defined USE_MIPMAP
    MipmapFieldComputation _fieldComputation;
#else
    BaselineFieldComputation _fieldComputation;
#endif
    TsneParameters _params;
    Bounds3D _bounds;
    utils::AbstractLog* _logger;

    TIMERS_DECLARE(
      TIMER_SUM_Q,
      TIMER_GRADIENTS,
      TIMER_UPDATE,
      TIMER_BOUNDS,
      TIMER_CENTERING
    )
  };
}

#endif