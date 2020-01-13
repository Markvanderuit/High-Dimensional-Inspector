
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <bitset>
#include <QVector3D>
#include <QTransform>
#include "depth_field_computation.h"
#include "depth_field_shaders.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/log_helper_functions.h"

// Do or do not add query timer functions
#ifdef FIELD_QUERY_TIMER_ENABLED
  #include <deque>
  #define startTimer(type) startTimerQuery(type)
  #define stopTimer() stopTimerQuery()
  #define updateTimers(iterations) updateTimerQueries(iterations)
  #define reportTimer(type, name, iteration) reportTimerQuery(type, name, iteration)
#else
  #define startTimer(type)
  #define stopTimer()
  #define updateTimers(iterations)
  #define reportTimer(type, name, iteration)
#endif // FIELD_QUERY_TIMER_ENABLED

namespace {
  // Magic numbers
  constexpr float pointSize = 2.f;
  constexpr int seedOffset = 7; // Generated by fair d20 die roll
  const float toDegMult = 180.f / M_PI;

  // For debug output
  void normalize(std::vector<float>& b) {
    const float max = *std::max_element(b.begin(), b.end());
    const float min = *std::min_element(b.begin(), b.end());
    const float range = max - min;
    std::transform(b.begin(), b.end(), b.begin(),
      [&range, &min](const auto& v) { return (v - min) / range; });
  }

  // Random number generation
  // Yeah yeah rand() is bad
  template <typename T>
  inline T next1D() {
    constexpr T next1DDiv = static_cast<T>(1) / static_cast<T>(RAND_MAX + 1);
    return static_cast<T>(rand()) * next1DDiv;
  }
}

namespace hdi::dr {
  DepthFieldComputation::DepthFieldComputation()
  : _initialized(false), _iteration(0), _w(0), _h(0), _d(0) {
    // ...
  }

  DepthFieldComputation::~DepthFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  void DepthFieldComputation::initialize(const TsneParameters& params, 
                                          unsigned n) {
    _params = params;

    // Initialize rng, yeah yeah std::rand() is bad.
    std::srand(_params._seed + seedOffset);

    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_DEPTH].addShader(VERTEX, depth_vert_src);
      _programs[PROGRAM_DEPTH].addShader(GEOMETRY, depth_geom_src);
      _programs[PROGRAM_GRID].addShader(VERTEX, depth_vert_src);
      _programs[PROGRAM_GRID].addShader(FRAGMENT, grid_fragment_src);
      _programs[PROGRAM_FIELD_3D].addShader(COMPUTE, field_src); 
      _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Fill _cellData for precomputed voxel grid cell data
    std::vector<uint32_t> bitData;
    for (int i = 31; i >= 0; i--) {
      // bitData.push_back(1u << i | (1u << std::min(31, i + 1)) | (1u << std::max(0, i - 1)));
      bitData.push_back(1u << i);
    }
    _cellData.fill(0u);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < bitData.size(); j++) {
        _cellData[i * 4 * bitData.size() + i + 4 * j] = bitData[j];
      }
    }

    #ifdef FIELD_QUERY_TIMER_ENABLED
      // Generate timer query handles and reset values
      glGenQueries(_timerHandles.size(), _timerHandles.data());
      _timerValues.fill({0l, 0l });
    #endif // FIELD_QUERY_TIMER_ENABLED

    glGenTextures(_textures.size(), _textures.data());
    glGenFramebuffers(_framebuffers.size(), _framebuffers.data());

    // Generate 2d depth texture array
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH_ARRAY]);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 2d depth texture
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DEPTH_SINGLE]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate voxel grid cell texture
    glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    // Generate voxel grid texture
    glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 3d fields texture
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate framebuffer for depth computation
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_DEPTH]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _textures[TEXTURE_DEPTH_ARRAY], 0);
    glDrawBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Generate framebuffer for grid computation and depth merging
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_GRID]);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[TEXTURE_GRID], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, _textures[TEXTURE_DEPTH_SINGLE], 0);
    glDrawBuffers(2, std::array<GLuint, 2> { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 }.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;
  }

  void DepthFieldComputation::clean() {
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(_framebuffers.size(), _framebuffers.data());
    glDeleteVertexArrays(1, &_point_vao);
#ifdef FIELD_QUERY_TIMER_ENABLED
    glDeleteQueries(_timerHandles.size(), _timerHandles.data());
#endif // FIELD_QUERY_TIMER_ENABLED
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
  }

#ifdef FIELD_ROTATE_VIEW
  void DepthFieldComputation::updateView() {
    constexpr float pi2 = 2.f * static_cast<float>(M_PI);
    const float r1 = next1D<float>(), r2 = next1D<float>(), r3 = next1D<float>();

    // Generate random point on unit sphere as direction
    float z = 1.f - 2.f * r1;
    float r = std::sqrtf(1.f - z * z);
    float phi = pi2 * r2;
    float x = std::cosf(phi) * r;
    float y = std::sinf(phi) * r;
    QVector3D a(x, y, z);
    a.normalize(); // JIC, should be fine though

    // Generate rotation towards direction vector
    QVector3D v(0, 0, 1);
    QVector3D b = QVector3D::crossProduct(v, a);
    float theta1 = std::acosf(QVector3D::dotProduct(v, a)) * toDegMult;
    
    // Generate random rotation around direction vector
    float theta2 = 360.f * r3;

    // Center point in [-0.5, 0.5], rotate, and uncenter again
    _viewMatrix = QMatrix4x4(); 
    _viewMatrix.translate(0.5f, 0.5f, 0.5f); // shift center back again
    _viewMatrix.rotate(theta2, a);
    _viewMatrix.rotate(theta1, b);
    _viewMatrix.translate(-0.5f, -0.5f, -0.5f); // shift center of [0, 1] embedding
  }
#endif // FIELD_ROTATE_VIEW

  void DepthFieldComputation::compute(unsigned w, unsigned h, unsigned d,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                Bounds3D bounds) {
#ifdef FIELD_ROTATE_VIEW
    // Generate new view matrix first
    updateView();
#endif // FIELD_ROTATE_VIEW
    
    QMatrix4x4 inverse = _viewMatrix.inverted();
    Point3D minBounds = bounds.min;
    Point3D range = bounds.range();

    // Rescale textures if necessary
    if (_w != w || _h != h || _d != d) {
      _w = w;
      _h = h;
      _d = d;

      // Update cell data for voxel grid computation only to d = 128
      glActiveTexture(GL_TEXTURE0);
      if (_d <= 128u) {
        glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI, _d, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, _cellData.data());
      }       
      glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH_ARRAY]);
      glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, _w, _h, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DEPTH_SINGLE]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, _w, _h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _w, _h, _d, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Compute depth textures
    { 
      startTimer(TIMER_DEPTH);

      // Bind bounds buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Bind uniforms to shader
      auto& program = _programs[PROGRAM_DEPTH];
      program.bind();
      program.uniformMatrix4f("transformation", _viewMatrix.data());
      
      // Prepare for drawing with depth test
      glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_DEPTH]);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glDepthMask(GL_TRUE);
      glViewport(0, 0, _w, _h);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
      // Draw a point at each embedding position
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(pointSize);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_DEPTH_TEST);
      _programs[PROGRAM_DEPTH].release();

      stopTimer();
    }

    // Compute grid texture and merge depth array texture in same shader
    {
      startTimer(TIMER_GRID);
      
      // Bind uniforms to program
      auto& program = _programs[PROGRAM_GRID];
      program.bind();
      program.uniformMatrix4f("transformation", _viewMatrix.data());
      program.uniform1i("depthMaps", 0);
      program.uniform1i("cellMap", 1);
      program.uniform1f("zPadding", 0.025f);

      // Bind the textures for this program
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH_ARRAY]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_1D, _textures[TEXTURE_CELLMAP]);

      // Bind bounds buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Prepare for drawing
      glBindFramebuffer(GL_FRAMEBUFFER, _framebuffers[FBO_GRID]);
      glViewport(0, 0, _w, _h);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glClear(GL_COLOR_BUFFER_BIT);
      glEnable(GL_COLOR_LOGIC_OP);
      glLogicOp(GL_OR);
      
      // Draw a point at each embedding position
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(pointSize);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_LOGIC_OP);
      program.release();
      stopTimer();
    }

    // Compute fields 3D texture
    {
      startTimer(TIMER_FIELD_3D);

      auto &program = _programs[PROGRAM_FIELD_3D];
      program.bind();

      // Attach depth texture for sampling
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DEPTH_SINGLE]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);

      // Set uniforms in compute shader
      program.uniform1ui("num_points", n);
      program.uniform3ui("texture_size", _w, _h, _d);
      program.uniformMatrix4f("invTransformation", inverse.data());
      program.uniform1i("depth_texture", 0);
      program.uniform1i("grid_texture", 1);

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_FIELD_3D], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(_w, _h, 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      
      program.release();
      stopTimer();
    }

    // Query field texture for positional values
    {
      glAssert("Querying field textures");
      startTimer(TIMER_INTERP);

      auto &program = _programs[PROGRAM_INTERP];
      program.bind();

      // Bind the textures for this shader
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_DEPTH_SINGLE]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);

      // Bind uniforms for this shader
      program.uniform1ui("num_points", n);
      program.uniform3ui("texture_size", _w, _h, _d);
      program.uniformMatrix4f("transformation", _viewMatrix.data());
      program.uniform1i("depth_texture", 0);
      program.uniform1i("fields_texture", 1);

      // Dispatch compute shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(128, 1, 1);

      // Cleanup
      program.release();
      stopTimer();
      glAssert("Queried field textures");
    }

    // Test output textures every 100 iterations
    #ifdef FIELD_IMAGE_OUTPUT
    if (_iteration == 999) {
      // Output grid
      {
        std::vector<uint32_t> buffer(_w * _h * 4, 0u);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _textures[TEXTURE_GRID]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, buffer.data());
        int w = _w / 2;
        for (int i = 0; i < _h; i++) {
          std::cout << std::bitset<32>(buffer[4 * i * _w + 4 * w]) << " "
                    << std::bitset<32>(buffer[4 * i * _w + 4 * w + 1]) << " "
                    << std::bitset<32>(buffer[4 * i * _w + 4 * w + 2]) << " "
                    << std::bitset<32>(buffer[4 * i * _w + 4 * w + 3]) << std::endl;
        }
        std::cout << std::endl;
        int c = 0;
        for (int i = 0; i < buffer.size(); i++) {
          c += std::bitset<32>(buffer[i]).count();
        }
        std::cout << "Dims of grid: " << _w << ", " << _h << ", " << _d << std::endl;
        std::cout << "Count of grid " << c  << std::endl << std::endl;
      }

      // Output field
      {
        std::vector<float> buffer(4 * _w * _h * _d);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
        glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, buffer.data());
        int w = _w / 2;
        for (int i = 0; i < _h; i++) {
          for (int j = 0; j < _d; j++) {
            int ind = j * (4 * _w * _h) + i * (4 * _w) + (4 * w);
            auto& v = buffer[ind];
            printf("%3.2f ", v);
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }
    // if (_iteration % 10 == 0) {
        // ...
    // }
    #endif // FIELD_IMAGE_OUTPUT
    
    updateTimers(_iteration);
    if (_iteration >= _params._iterations - 1) {
      // Output timer values
      reportTimer(TIMER_DEPTH, "Depth", _iteration);
      reportTimer(TIMER_GRID, "Grid", _iteration);
      reportTimer(TIMER_FIELD_3D, "Field", _iteration);
      reportTimer(TIMER_INTERP, "Interp", _iteration);
      utils::secureLog(_logger, "");
    }
    _iteration++;
  }

#ifdef FIELD_QUERY_TIMER_ENABLED
  void DepthFieldComputation::startTimerQuery(TimerType type) {
    glBeginQuery(GL_TIME_ELAPSED, _timerHandles[type]);
  }

  void DepthFieldComputation::stopTimerQuery() {
    glEndQuery(GL_TIME_ELAPSED);
  }
  
  bool DepthFieldComputation::updateTimerQuery(TimerType type, unsigned iteration) {
    const auto& handle = _timerHandles[type];
    
    // Return if query data for this handle is unavailable
    int i = 0;
    glGetQueryObjectiv(handle, GL_QUERY_RESULT_AVAILABLE, &i);
    if (!i) {
      return false;
    }
    
    // Obtain query value
    auto& values = _timerValues[type];
    glGetQueryObjecti64v(handle, GL_QUERY_RESULT, &values[TIMER_LAST_QUERY]);
    
    // Compute incremental times
    values[TIMER_TOTAL] += values[TIMER_LAST_QUERY];
    values[TIMER_AVERAGE] 
      = values[TIMER_AVERAGE] 
      + (values[TIMER_LAST_QUERY] - values[TIMER_AVERAGE]) 
      / (iteration + 1);

    return true;
  }

  void DepthFieldComputation::updateTimerQueries(unsigned iteration) {
    std::deque<int> types(TimerTypeLength);
    std::iota(types.begin(), types.end(), 0);

    // Query until all timer data is available and obtained
    while (!types.empty()) {
      const auto type = types.front();
      types.pop_front();
      if (!updateTimerQuery(TimerType(type), iteration)) {
        types.push_back(type);
      }
    }
  }

  void DepthFieldComputation::reportTimerQuery(TimerType type, const std::string& name, unsigned iteration) {
    const double averageMs = static_cast<double>(_timerValues[type][TIMER_AVERAGE]) / 1000000.0;
    const double totalMs = static_cast<double>(_timerValues[type][TIMER_TOTAL]) / 1000000.0;
    const std::string value = std::to_string(averageMs) + ", " + std::to_string(totalMs);
    utils::secureLogValue(
      _logger, 
      name + " shader average, total (ms)", 
      value);
  }
#endif // FIELD_QUERY_TIMER_ENABLED
}