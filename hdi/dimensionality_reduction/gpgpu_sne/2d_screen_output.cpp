#define _USE_MATH_DEFINES
#include <algorithm>
#include <execution>
#include <cmath>
#include "2d_screen_output.h"
#include "2d_screen_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/visual_utils.h"

constexpr uint reduceWorkGroupWidth = 16u;
constexpr uint reduceIterValueSize = 8 * sizeof(float);

namespace hdi::dr {
  Screen2dOutput::Screen2dOutput()
  : _initialized(false),
    _logger(nullptr)
  { }

  Screen2dOutput::~Screen2dOutput()
  {
    if (_initialized) {
      clean();
    }
  }

  void Screen2dOutput::initialize(const TsneParameters& params)
  {
    TIMERS_CREATE();
    _params = params;

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_REDUCE].addShader(COMPUTE, reduce_src);
      _programs[PROGRAM_NORMALIZE].addShader(COMPUTE, normalize_src);
      _programs[PROGRAM_SCREEN].addShader(VERTEX, screen_vert_src);
      _programs[PROGRAM_SCREEN].addShader(FRAGMENT, screen_frag_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // not recovering from incorrect shaders
    }

    // Initialize buffers
    glCreateBuffers(_buffers.size(), _buffers.data());
    glNamedBufferData(_buffers[BUFFER_REDUCE_1], reduceIterValueSize, nullptr, GL_DYNAMIC_READ);

    // Initialize textures
    glGenTextures(_textures.size(), _textures.data());
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_NORMALIZED]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    // Initialize buffer with quad vertex data loaded
    std::array<GLfloat, 8> quad = {
      -1.f, 1.f,
      -1.f, -1.f,
      1.f, 1.f,
      1.f, -1.f
    };
    glNamedBufferStorage(_buffers[BUFFER_QUAD],
                         quad.size() * sizeof(GLfloat),
                         quad.data(),
                         GL_DYNAMIC_STORAGE_BIT);

    // Initialize vertex array
    glCreateVertexArrays(1, &vao);
    glEnableVertexArrayAttrib(vao, 0); 
    glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayVertexBuffer(vao, 0, _buffers[BUFFER_QUAD], 0, 2 * sizeof(GLfloat)); // check 2
    glVertexArrayAttribBinding(vao, 0, 0);

    GL_ASSERT("Screen2dOutput::initialize()");
    _initialized = true;
  }

  void Screen2dOutput::clean()
  {
    TIMERS_DESTROY();
    for (auto& program : _programs) {
      program.destroy();
    }
    _programs.fill(ShaderProgram());
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTextures(_textures.size(), _textures.data());
    _initialized = false;
  }

  void Screen2dOutput::compute(GLuint textureHandle, 
                               unsigned w, 
                               unsigned h, 
                               unsigned iter)
  {
    if (_w != w || _h != h) {
      _w = w;
      _h = h;

      const unsigned bufferSize = ceilDiv(_w, reduceWorkGroupWidth) 
                                * ceilDiv(_h, reduceWorkGroupWidth) 
                                * reduceIterValueSize;
      glNamedBufferData(_buffers[BUFFER_REDUCE_0], bufferSize, nullptr, GL_DYNAMIC_DRAW);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_NORMALIZED]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Check if a texture is provided
    if (glIsTexture(textureHandle) == GL_FALSE) {
      return;
    }

    // Reduce texture to determine maximum value
    {
      TIMER_TICK(TIMER_REDUCE)
      auto& program = _programs[PROGRAM_REDUCE];
      program.bind();

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_REDUCE_0]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_REDUCE_1]);

      // Bind texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, textureHandle);

      // Set uniforms
      program.uniform1i("textureSampler", 0);
      program.uniform2i("textureSize", _w, _h);

      // First iteration, reduce add texture into BUFFER_REDUCE_0
      {
        program.uniform1ui("iteration", 0);

        glDispatchCompute(ceilDiv(_w, reduceWorkGroupWidth), 
                          ceilDiv(_h, reduceWorkGroupWidth), 
                          1u);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      // Second iteration, reduce add BUFFER_REDUCE_0 into BUFFER_REDUCE_1
      {
        const unsigned bufferSize = ceilDiv(_w, reduceWorkGroupWidth) 
                                  * ceilDiv(_h, reduceWorkGroupWidth);
        program.uniform1ui("iteration", 1);
        program.uniform1ui("reduceIter0Size", bufferSize);

        glDispatchCompute(1u, 1u, 1u);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
      
      program.release();
      TIMER_TOCK(TIMER_REDUCE)
    }

    // Normalize texture
    {
      TIMER_TICK(TIMER_NORMALIZE)
      auto& program = _programs[PROGRAM_NORMALIZE];
      program.bind();

      // Bind buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_REDUCE_1]);
      
      // Bind texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, textureHandle);

      // Set uniforms
      program.uniform1i("textureSampler", 0);
      program.uniform2i("textureSize", _w, _h);

      // Bind image
      glBindImageTexture(0, _textures[TEXTURE_NORMALIZED], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Run prg
      glDispatchCompute(ceilDiv(_w, 16u), ceilDiv(_h, 16u), 1u);

      program.release();
      TIMER_TOCK(TIMER_NORMALIZE)
    }

    // Draw to screen
    {
      TIMER_TICK(TIMER_SCREEN)
      auto& program = _programs[PROGRAM_SCREEN];
      program.bind();

      // Set uniforms
      program.uniform1i("textureSampler", 0);

      // Bind texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_NORMALIZED]);

      // Set viewport
      glViewport(0, 0, 1920, 1920);

      // Set framebuffer
      GLuint fbo = 0; // use default framebuffer
      constexpr std::array<GLfloat, 6> rgba = {
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f
      };
      constexpr float depth = 0.f;
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glClearNamedFramebufferfv(fbo, GL_COLOR, 0, rgba.data());
      glClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &depth);

      // Draw quad
      glBindVertexArray(vao);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glBindVertexArray(0);
      
      program.release();
      TIMER_TOCK(TIMER_SCREEN)
    }

    // Update timers, log values on final iteration
    TIMERS_UPDATE()
    if (TIMERS_LOG_ENABLE && iter >= _params._iterations - 1) {
      utils::secureLog(_logger, "Screen output");
      TIMER_LOG(_logger, TIMER_REDUCE, "  Reduce")
      TIMER_LOG(_logger, TIMER_NORMALIZE, "  Normal")
      TIMER_LOG(_logger, TIMER_SCREEN, "  Screen")
      utils::secureLog(_logger, "");
    }
  }
}