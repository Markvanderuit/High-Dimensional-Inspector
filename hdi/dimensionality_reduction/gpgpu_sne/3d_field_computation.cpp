#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <cstdint>
#include "3d_field_computation.h"
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

#define GLSL(name, version, shader) \
  static const char* name = \
  "#version " #version "\n" #shader

namespace {
  // Compute shader for 3d stencil texture generation
  GLSL(stencil_src, 430,
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    layout(r8, binding = 0) writeonly uniform image3D stencil_texture;

    uniform uint num_points;

    void main() {
      for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
            i < num_points;
            i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        // Get the point position in texture
        vec3 pos = imageSize(stencil_texture) * (Positions[i] - minBounds) * invRange;
        ivec3 minp = ivec3(floor(pos));
        ivec3 maxp = ivec3(ceil(pos));

        // fill 9 nearest neighboring pixels (not exactly clever code)
        imageStore(stencil_texture, ivec3(minp.x, minp.y, minp.z), vec4(1));
        imageStore(stencil_texture, ivec3(minp.x, minp.y, maxp.z), vec4(1));
        imageStore(stencil_texture, ivec3(minp.x, maxp.y, minp.z), vec4(1));
        imageStore(stencil_texture, ivec3(minp.x, maxp.y, maxp.z), vec4(1));
        imageStore(stencil_texture, ivec3(maxp.x, minp.y, minp.z), vec4(1));
        imageStore(stencil_texture, ivec3(maxp.x, minp.y, maxp.z), vec4(1));
        imageStore(stencil_texture, ivec3(maxp.x, maxp.y, minp.z), vec4(1));
        imageStore(stencil_texture, ivec3(maxp.x, maxp.y, maxp.z), vec4(1));
      }
    }
  );

  // Compute shader for 3d fields texture
  // r = density, g = x gradient, b = y gradient, a = z gradient
  GLSL(fields_src, 430,
    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    layout(rgba32f, binding = 0) writeonly uniform image3D fields_texture;
    layout(r8, binding = 1) readonly uniform image3D stencil_texture;

    uniform uint num_points;
    uniform uvec3 texture_size;
    uniform float function_support;

    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec4 reductionArray[halfGroupSize];

    void main() {
      ivec3 xyz = ivec3(gl_WorkGroupID);

      // Test stencil texture to maybe ignore pixel
      if (imageLoad(stencil_texture, xyz).x == 0) {
        return;
      }

      // Invocation location
      uint lid = gl_LocalInvocationIndex.x;

      // Compute pixel position in domain space
      vec3 pos = ((vec3(xyz) + vec3(0.5)) / texture_size) * range + minBounds;
      
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        // Compute S = t_stud and V = t_stud_2
        vec3 t = pos - Positions[i];
        float t_stud = 1.f / (1.f + dot(t, t));
        vec3 t_stud_2 = t * (t_stud * t_stud);
        
        // 3d field layout is: S, V.x, V.y, V.z
          v += vec4(t_stud, t_stud_2);
      }

      // Perform reduce add over all computed points for this pixel
      if (lid >= halfGroupSize) {
        reductionArray[lid - halfGroupSize] = v;
      }
      barrier();
      if (lid < halfGroupSize) {
        reductionArray[lid] += v;
      }
      for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
        barrier();
        if (lid < i) {
          reductionArray[lid] += reductionArray[lid + i];
        }
      }
      barrier();
      if (lid < 1) {
        vec4 reducedArray = reductionArray[0] + reductionArray[1];
        imageStore(fields_texture, xyz, reducedArray);
      }
    }
  );

  GLSL(interp_src, 430, 
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
    layout(std430, binding = 1) buffer Val { vec4 Values[]; };
    layout(std430, binding = 2) buffer BoundsInterface { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };

    uniform uint num_points;
    uniform sampler3D fields_texture;

    void main() {
      // Grid stride loop, straight from CUDA, scales better for very large N
      for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
          i < num_points;
          i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        // Map position of point to [0, 1]
        vec3 position = (Positions[i] - minBounds) * invRange;

        // Sample texture at position
        Values[i] = texture(fields_texture, position);
      }
    }
  );
}

namespace hdi::dr {
  Compute3DFieldComputation::Compute3DFieldComputation()
  : _initialized(false), _iteration(0), _w(0), _h(0), _d(0) {
    // ...
  }

  Compute3DFieldComputation::~Compute3DFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  void Compute3DFieldComputation::initialize(const TsneParameters& params, 
                                            unsigned n) {
    _params = params;

    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_STENCIL].addShader(COMPUTE, stencil_src);
      _programs[PROGRAM_FIELD_3D].addShader(COMPUTE, fields_src);
      _programs[PROGRAM_INTERP].addShader(COMPUTE, interp_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    #ifdef FIELD_QUERY_TIMER_ENABLED
      // Generate timer query handles and reset values
      glGenQueries(_timerHandles.size(), _timerHandles.data());
      _timerValues.fill({0l, 0l });
    #endif // FIELD_QUERY_TIMER_ENABLED

    glGenTextures(_textures.size(), _textures.data());

    // Generate 3d stencil texture
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_STENCIL]);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Generate 3d fields texture
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    _iteration = 0;
    _initialized = true;
  }

  void Compute3DFieldComputation::clean() {
#ifdef FIELD_QUERY_TIMER_ENABLED
    glDeleteQueries(_timerHandles.size(), _timerHandles.data());
#endif // FIELD_QUERY_TIMER_ENABLED
    glDeleteTextures(_textures.size(), _textures.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
  }

  void Compute3DFieldComputation::compute(unsigned w, unsigned h, unsigned d,
                  float function_support, unsigned n,
                  GLuint position_buff, GLuint bounds_buff, GLuint interp_buff,
                  Bounds3D bounds) {
    // Only update texture dimensions if they have changed
    bool refresh = false;
    if (_w != w || _h != h || _d != d) {
      refresh = true;
      _w = w;
      _h = h;
      _d = d;
    }

    // Compute 3d stencil texture
    {
      startTimer(TIMER_STENCIL);
      auto& program = _programs[PROGRAM_STENCIL];
      program.bind();
      program.uniform1ui("num_points", n);

      // Define stencil texture format
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_STENCIL]);
      if (refresh) {
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, w, h, d, 0, GL_RED, GL_FLOAT, nullptr);
      }

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_STENCIL], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

      // Perform computation over groups of points
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      stopTimer();
    }

    // Compute 3d fields texture
    {
      startTimer(TIMER_FIELD_3D);
      auto& program = _programs[PROGRAM_FIELD_3D];
      program.bind();
      program.uniform1ui("num_points", n);
      program.uniform1f("function_support", function_support);

      // Define fields texture format
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
      if (refresh) {
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, w, h, d, 0, GL_RGBA, GL_FLOAT, nullptr);
        program.uniform3ui("texture_size", w, h, d);
      }

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_FIELD_3D], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      glBindImageTexture(1, _textures[TEXTURE_STENCIL], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R8);

      // Perform computation over every pixel
      glDispatchCompute(w, h, d);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

      program.release();
      stopTimer();
    }

    // Query field texture for positional values
    {
      startTimer(TIMER_INTERP);
      auto& program = _programs[PROGRAM_INTERP];
      program.bind();

      // Bind the textures for this shader
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, interp_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bounds_buff);
      
      // Bind uniforms for this shader
      program.uniform1ui("num_points", n);
      program.uniform1i("fields_texture", 0);

      // Dispatch compute shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(128, 1, 1);

      program.release();
      stopTimer();
    }

    updateTimers(_iteration);
    if (_iteration >= _params._iterations - 1) {
      // Output timer values
      reportTimer(TIMER_STENCIL, "Stencil", _iteration);
      reportTimer(TIMER_FIELD_3D, "Field", _iteration);
      reportTimer(TIMER_INTERP, "Interp", _iteration);
    }
    _iteration++;
  }

#ifdef FIELD_QUERY_TIMER_ENABLED
  void Compute3DFieldComputation::startTimerQuery(TimerType type) {
    glBeginQuery(GL_TIME_ELAPSED, _timerHandles[type]);
  }

  void Compute3DFieldComputation::stopTimerQuery() {
    glEndQuery(GL_TIME_ELAPSED);
  }
  
  bool Compute3DFieldComputation::updateTimerQuery(TimerType type, unsigned iteration) {
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
    
    // Compute incremental average
    values[TIMER_AVERAGE] 
      = values[TIMER_AVERAGE] 
      + (values[TIMER_LAST_QUERY] - values[TIMER_AVERAGE]) 
      / (iteration + 1);

    return true;
  }

  void Compute3DFieldComputation::updateTimerQueries(unsigned iteration) {
    std::deque<TimerType> types = {
      TIMER_STENCIL,
      TIMER_FIELD_3D,
      TIMER_INTERP
    };

    // Query until all timer data is available and obtained
    while (!types.empty()) {
      const auto type = types.front();
      types.pop_front();
      if (!updateTimerQuery(type, iteration)) {
        types.push_back(type);
      }
    }
  }

  void Compute3DFieldComputation::reportTimerQuery(TimerType type, const std::string& name, unsigned iteration) {
    const double ms = static_cast<double>(_timerValues[type][TIMER_AVERAGE]) / 1000000.0;
    utils::secureLogValue(
      _logger, 
      name + " shader average (" + std::to_string(iteration) + " iters, ms)",
      ms);
  }
#endif // FIELD_QUERY_TIMER_ENABLED
}