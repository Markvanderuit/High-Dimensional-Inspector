#pragma once

#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "gpgpu_utils.h"
#include "3d_utils.h"

// #define FIELD_IMAGE_OUTPUT // Output field images every 100 iterations

namespace hdi::dr {
  class BaselineFieldComputation {
  public:
    BaselineFieldComputation();
    ~BaselineFieldComputation();

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

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }
  private:
    bool _initialized;
    int _iteration;
    unsigned _w, _h, _d;

    enum TextureType {
      // Enum values matching to textures in _textures array
      TEXTURE_CELLMAP,
      TEXTURE_GRID,
      TEXTURE_FIELD_3D,

      // Static enum length
      TextureTypeLength 
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_GRID,
      PROGRAM_FIELD_3D,
      PROGRAM_INTERP,

      // Static enum length
      ProgramTypeLength
    };

    enum FramebufferType {
      // Enum values matching to framebuffers programs in _framebuffers array
      FBO_GRID,

      // Static enum length
      FramebufferTypeLength
    };

    GLuint _point_vao;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, FramebufferTypeLength> _framebuffers;
    std::array<GLuint, TextureTypeLength> _textures;
    std::array<uint32_t, 4 * 128> _cellData;
    TsneParameters _params;
    utils::AbstractLog* _logger;
    
    // Query timers matching to each shader
    TIMERS_DECLARE(
      GRID, 
      FIELD_3D, 
      INTERP
    )
  };
}