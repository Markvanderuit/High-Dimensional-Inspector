#ifndef __APPLE

#pragma once

#include <array>
#include <QMatrix4x4>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "3d_utils.h"

#define FIELD_QUERY_TIMER_ENABLED // Enable GL timer queries and output info on final iteration
// #define FIELD_IMAGE_OUTPUT // Output field images every 100 iterations

namespace hdi::dr {
  class DepthFieldComputation {
  public:
    DepthFieldComputation();
    ~DepthFieldComputation();

    // Initialize gpu components for computation
    void initialize(const TsneParameters& params,
                    unsigned n);

    // Remove gpu components
    void clean();

    // Compute field and depth textures
    void compute(unsigned w, unsigned h, unsigned d,
                 float function_support, unsigned n,
                 GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                 Bounds3D bounds);

  private:
    void updateView();

    bool _initialized;
    int _iteration;
    unsigned _w, _h, _d;

    enum TextureType {
      // Enum values matching to textures in _textures array
      TEXTURE_DEPTH_ARRAY,
      TEXTURE_DEPTH_SINGLE,
      TEXTURE_CELLMAP,
      TEXTURE_GRID,
      TEXTURE_FIELD_3D,
      TEXTURE_DEBUG,

      // Static enum length
      TextureTypeLength 
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_DEPTH,
      PROGRAM_GRID,
      // PROGRAM_DENSITY_ESTIMATION,
      PROGRAM_FIELD_3D,
      PROGRAM_INTERP,
      PROGRAM_DEBUG,

      // Static enum length
      ProgramTypeLength
    };

    enum FramebufferType {
      // Enum values matching to framebuffers programs in _framebuffers array
      FBO_DEPTH,
      FBO_GRID,

      // Static enum length
      FramebufferTypeLength
    };

    GLuint _point_vao;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, FramebufferTypeLength> _framebuffers;
    std::array<GLuint, TextureTypeLength> _textures;
    std::array<uint32_t, 4 * 128> _cellData;
    QMatrix4x4 _viewMatrix;
    TsneParameters _params;
    utils::AbstractLog* _logger;

#ifdef FIELD_QUERY_TIMER_ENABLED
  private:
    enum TimerType {
      TIMER_DEPTH,
      TIMER_GRID,
      TIMER_FIELD_3D,
      TIMER_INTERP,

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
    bool updateTimerQuery(TimerType type, unsigned iteration);
    void updateTimerQueries(unsigned iteration);
    void reportTimerQuery(TimerType type, const std::string& name, unsigned iteration);
    std::array<GLuint, TimerTypeLength> _timerHandles;
    std::array<std::array<GLint64, TimerValueLength>, TimerTypeLength> _timerValues;
#endif // FIELD_QUERY_TIMER_ENABLED

  public:
    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }
  };
}

#endif