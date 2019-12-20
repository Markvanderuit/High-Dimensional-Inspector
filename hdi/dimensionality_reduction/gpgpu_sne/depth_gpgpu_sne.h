#ifndef __APPLE__

#pragma once

#include <array>
#include <QMatrix4x4>
#include "hdi/utils/abstract_log.h"
#include "hdi/utils/glad/glad.h"
#include "hdi/data/shader.h"
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "3d_utils.h"
#include "field_depth_computation.h"

#define QUERY_TIMER_ENABLED // Enable GL timer queries and output info on final iteration

namespace hdi::dr {
  class GpgpuDepthSneCompute {
  public:
    // Base constr., destr.
    GpgpuDepthSneCompute();
    ~GpgpuDepthSneCompute();

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
    }

  private:
    void computeBounds(unsigned n, float padding);
    void updateView();
    void interpolateFields(unsigned n, float* sum_Q);
    void sumQ(unsigned in, float* sum_Q);
    void computeGradients(unsigned n, float sum_Q, float exaggeration);
    void updatePoints(unsigned n, unsigned iteration, float mult);
    void updateEmbedding(unsigned n, float exaggeration, unsigned iteration);

    bool _initialized;
    bool _adaptive_resolution;
    float _resolution_scaling;
    
    enum BufferType {
      // Enums matching to storage buffers in _buffers array
      BUFFER_POSITION,
      BUFFER_INTERP_FIELDS,
      BUFFER_SUM_Q,
      BUFFER_NEIGHBOUR,
      BUFFER_PROBABILITIES,
      BUFFER_INDEX,
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
      PROGRAM_INTERP_FIELDS,
      PROGRAM_SUM_Q,
      PROGRAM_FORCES,
      PROGRAM_UPDATE,
      PROGRAM_BOUNDS,
      PROGRAM_CENTERING,
      
      // Static enum length
      ProgramTypeLength 
    };
    
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    DepthFieldComputation _depthFieldComputation;
    TsneParameters _params;
    Bounds3D _bounds;
    QMatrix4x4 viewMatrix;
    utils::AbstractLog* _logger;
  
#ifdef QUERY_TIMER_ENABLED
  private:
    enum TimerType {
      // Enums matching to timers 
      TIMER_INTERP_FIELDS,
      TIMER_SUM_Q,
      TIMER_FORCES,
      TIMER_UPDATE,
      TIMER_BOUNDS,
      TIMER_CENTERING,
      
      // Static enum length
      TimerTypeLength 
    };
    enum TimerValue {
      TIMER_LAST_QUERY,
      TIMER_AVERAGE,

      // Static enum length
      TimerValueLength 
    };

    void startTimerQuery(TimerType type);
    void stopTimerQuery();
    void updateTimerQuery(TimerType type, unsigned iteration);
    void reportTimerQuery(TimerType type, const std::string& name, unsigned iteration);
    double getElapsed();
    std::array<GLuint, TimerTypeLength> _timerHandles;
    std::array<std::array<GLint64, TimerValueLength>, TimerTypeLength> _timerValues;
#endif // QUERY_TIMER_ENABLED
  };
}

#endif