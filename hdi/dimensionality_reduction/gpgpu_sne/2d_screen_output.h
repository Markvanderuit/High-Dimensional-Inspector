#pragma once

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "gpgpu_utils.h"
#include "2d_utils.h"

namespace hdi::dr {
  class Screen2dOutput {
  public:
    Screen2dOutput();
    ~Screen2dOutput();

    void initialize(const TsneParameters& params);
    void clean();
    void compute(GLuint textureHandle, 
                 unsigned w, 
                 unsigned h, 
                 unsigned iter);

    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }

  private:
    enum BufferType {
      // Enum values matching to buffers in _buffers array
      BUFFER_QUAD,
      BUFFER_REDUCE_0,
      BUFFER_REDUCE_1,

      // Static enum length
      BufferTypeLength
    };

    enum ProgramType {
      // Enum values matching to shader programs in _programs array
      PROGRAM_REDUCE,
      PROGRAM_NORMALIZE,
      PROGRAM_SCREEN,

      // Static enum length
      ProgramTypeLength
    };

    enum TextureType {
      // Enum values matching to texture handles in _textures array
      TEXTURE_NORMALIZED,

      // Static enum length
      TextureTypeLength
    };

    // Declare query timers matching to each shader
    TIMERS_DECLARE(
      TIMER_REDUCE, 
      TIMER_NORMALIZE,
      TIMER_SCREEN
    )

    bool _initialized;
    unsigned _w, _h;
    GLuint vao;
    utils::AbstractLog* _logger;
    TsneParameters _params;
    std::array<GLuint, BufferTypeLength> _buffers;
    std::array<ShaderProgram, ProgramTypeLength> _programs;
    std::array<GLuint, TextureTypeLength> _textures;
  };
}