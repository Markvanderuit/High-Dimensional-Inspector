#ifndef __APPLE

#pragma once

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "3d_utils.h"

#define FIELD_QUERY_TIMER_ENABLED // Enable GL timer queries and output info on final iteration

namespace hdi::dr {
  class Compute3DFieldComputation {
  public:
    Compute3DFieldComputation();
    ~Compute3DFieldComputation();

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
    bool _initialized;
    int _iteration;
    unsigned _w, _h, _d;

    enum TextureType {
      // Enum values matching to textures in _textures array
      TEXTURE_STENCIL,
      TEXTURE_FIELD_3D,

      // Static enum length
      TextureTypeLength 
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_STENCIL,
      PROGRAM_FIELD_3D,
      PROGRAM_INTERP,

      // Static enum length
      ProgramTypeLength
    };

    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, TextureTypeLength> _textures;
    TsneParameters _params;
    utils::AbstractLog* _logger;

#ifdef FIELD_QUERY_TIMER_ENABLED
  private:
    enum TimerType {
      TIMER_STENCIL,
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

#endif // _APPLE