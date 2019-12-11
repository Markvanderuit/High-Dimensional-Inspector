#include "field_depth_computation.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include <algorithm>
#include <QMatrix4x4>

#define GLSL(version, shader)  "#version " #version "\n" #shader
#define FIELD_IMAGE_OUTPUT // Output field images every 100 iterations
// #define STENCIL_TEST // Perform stencil test

namespace {
  // Magic numbers
  constexpr float pointSize = 3.0f;

  // For debug output
  void normalize(std::vector<float>& b) {
    const float max = *std::max_element(b.begin(), b.end());
    const float min = *std::min_element(b.begin(), b.end());
    const float range = max - min;
    std::transform(b.begin(), b.end(), b.begin(),
      [&range, &min](const auto& v) { return (v - min) / range; });
  }

  const char* stencil_vert_src = GLSL(430,
    layout(location = 0) in vec4 point;
    
    uniform vec3 min_bounds;
    uniform vec3 range;
    uniform mat4 rotation;

    vec3 removeZero(in vec3 v) {
      if (v.x == 0.f) v.x = 1.f;
      if (v.y == 0.f) v.y = 1.f;
      if (v.z == 0.f) v.z = 1.f;
      return v;
    }

    void main() {
      // const vec3 min_bounds = Bounds[0];
      // const vec3 range = Bounds[1] - min_bounds;
      // Transform point into [0, 1] space
      vec3 cRange = removeZero(range);
      vec3 position = (point.xyz - min_bounds) / cRange;
      // Apply transform
      position = vec3(rotation * vec4(position, 1));
      // Transform point into frustrum space
      gl_Position = vec4(position * 2.f - 1.f, 1.f);
      // gl_Position = vec4(((point.xyz - min_bounds) / range) * 2.f - 1.f, 1.f);
    }
  );

  const char* stencil_frag_src = GLSL(430,
    out vec2 fragColor;

    void main() {
      // x marks stencil, y marks depth in range [0, 1]
      fragColor = vec2(1.f, gl_FragCoord.z);
    }
  );

  const char* field_src = GLSL(430,
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    layout(std430, binding = 0) buffer PosInterface { vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { vec3 Bounds[]; };
    layout(rg32f, binding = 0) readonly uniform image2D stencil_depth_texture;
    layout(rgba32f, binding = 1) writeonly uniform image2D fields_texture;

    uniform uint num_points;
    uniform uvec2 texture_size;
    uniform float function_support;
    uniform mat4 invRotation;

    // Reduction components
    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec4 reductionArray[halfGroupSize];

    vec3 removeZero(in vec3 v) {
      if (v.x == 0.f) v.x = 1.f;
      if (v.y == 0.f) v.y = 1.f;
      if (v.z == 0.f) v.z = 1.f;
      return v;
    }

    void main() {
      // Pixels of current workgroup
      uint x = gl_WorkGroupID.x;
      uint y = gl_WorkGroupID.y;

      // ID of current local instance
      uint lid = gl_LocalInvocationIndex.x;

      // Get stencil/depth value
      vec2 stencil_depth = imageLoad(stencil_depth_texture, ivec2(x, y)).xy;
#ifdef STENCIL_TEST
      if (stencil_depth.x == 0.f) {
        if (lid == 0) {
          imageStore(fields_texture, ivec2(x, y), vec4(0));
        }
        return;
      }
#endif // STENCIL_TEST

      // Obtain bounds
      vec3 min_bounds = Bounds[0];
      vec3 max_bounds = Bounds[1];
      vec3 range = removeZero(max_bounds - min_bounds);

      // Domain pos uses closest measured depth for determining z
      vec3 pixel_pos = vec3((vec2(x, y) + vec2(0.5)) / texture_size, stencil_depth.y);
      // vec3 domain_pos = pixel_pos;
      vec3 domain_pos = vec3(invRotation * vec4(pixel_pos, 1));
      domain_pos = domain_pos * range + min_bounds;

      // Iterate over points to obtain density/gradient
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        vec3 t = domain_pos - Positions[i];
        float eucl_2 = dot(t, t);
        float t_stud = 1.f / (1.f + eucl_2);
        float t_stud_2 = t_stud * t_stud;

        // Field layout is: S, V.x, V.y, V.z
        v += vec4(t_stud, t_stud_2 * t.x, t_stud_2 * t.y, t_stud_2 * t.z);
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
        imageStore(fields_texture, ivec2(x, y), reducedArray);
      }
    }
  );
}

namespace hdi::dr {
  DepthFieldComputation::DepthFieldComputation() {
    // ...
  }

  DepthFieldComputation::~DepthFieldComputation() {
    if (_initialized) {
      clean();
    }
  }

  void DepthFieldComputation::initialize(unsigned n) {
    // Build shader programs
    try {
      _stencil_program.create();
      _stencil_program.addShader(VERTEX, stencil_vert_src);
      _stencil_program.addShader(FRAGMENT, stencil_frag_src);
      _stencil_program.build();
      _field_program.create();
      _field_program.addShader(COMPUTE, field_src);
      _field_program.build();
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Generate 2d stencil + depth texture
    glGenTextures(1, &_textures[STENCIL_DEPTH]);
    glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 2d fields texture
    glGenTextures(1, &_textures[FIELD]);
    glBindTexture(GL_TEXTURE_2D, _textures[FIELD]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate depth buffer for stencil/depth pass
    glGenRenderbuffers(1, &_depth_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1, 1);

    // Generate framebuffer for stencil/depth pass
    glGenFramebuffers(1, &_stencil_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _stencil_fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[STENCIL_DEPTH], 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depth_buffer);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;
  }

  void DepthFieldComputation::clean() {
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_stencil_fbo);
    glDeleteVertexArrays(1, &_point_vao);
    _stencil_program.destroy();
    _field_program.destroy();
    _iteration = 0;
    _initialized = false;
  }

  void DepthFieldComputation::compute(unsigned w, unsigned h,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff,
                Bounds3D bounds, QMatrix4x4 view) {
    // Inverse view matrix for reversing transformation
    QMatrix4x4 inverse = view.inverted();
    Point3D minBounds = bounds.min;
    Point3D range = bounds.range();

    // Compute stencil and depth texture
    { 
      // Configure framebuffer texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0, GL_RG, GL_FLOAT, nullptr);

      // Configure renderbuffer
      glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);

      // Bind uniforms to shader
      _stencil_program.bind();
      _stencil_program.uniform3f("min_bounds", minBounds.x, minBounds.y, minBounds.z);
      _stencil_program.uniform3f("range", range.x, range.y, range.z);
      _stencil_program.uniformMatrix4f("rotation", view.data());
      
      // Prepare for drawing with depth test
      glViewport(0, 0, w, h);
      glBindFramebuffer(GL_FRAMEBUFFER, _stencil_fbo);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glDepthMask(GL_TRUE);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
      // Draw a point at each embedding position
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(pointSize);
      // glEnable(GL_POINT_SMOOTH);
      glDrawArrays(GL_POINTS, 0, n);

      // Cleanup
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_DEPTH_TEST);
      _stencil_program.release();
    }

    // Compute field texture
    {
      _field_program.bind();

      // Define fields texture format
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, _textures[FIELD]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);

      // Set uniforms in compute shader
      _field_program.uniform1ui("num_points", n);
      _field_program.uniform2ui("texture_size", w, h);
      _field_program.uniform1f("function_support", function_support);
      _field_program.uniformMatrix4f("invRotation", inverse.data());

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[STENCIL_DEPTH], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
      glBindImageTexture(1, _textures[FIELD], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(w, h, 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      
      // Cleanup
      _field_program.release();
    }

    // Test output textures every 100 iterations
    #ifdef FIELD_IMAGE_OUTPUT
    if (_iteration % 100 == 0) {
      // Output depth/stencil
      {
        std::vector<float> stencil_depth(w * h * 2, 0.f);
        glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, stencil_depth.data());
        const std::array<std::string, 2> names = { 
          "images/stencil_",
          "images/depth_" 
        };

        float min = std::numeric_limits<float>::max();
        float max = -std::numeric_limits<float>::max();
        for (int i = 0; i < w * h; i++) {
          if (stencil_depth[2 * i + 0] != 0.f) {
            min = std::min(min, stencil_depth[2 * i + 1]);
            max = std::max(max, stencil_depth[2 * i + 1]);
          }
        }
        std::vector<float> buffer(w * h, 0.f);
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < w * h; j++) {
            buffer[j] = stencil_depth[2 * j + i];
          }
          // normalize(buffer);
          hdi::utils::valuesToImage(names[i] + std::to_string(_iteration), buffer, w, h, 1);
        }
      }

      // Output fields
      {
        std::vector<float> field(w * h * 4, 0.f);
        glBindTexture(GL_TEXTURE_2D, _textures[FIELD]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, field.data());
      
        const std::array<std::string, 4> names = { 
          "images/density_",
          "images/gradient_x_",
          "images/gradient_y_",
          "images/gradient_z_" 
        };

        std::vector<float> buffer(w * h, 0.f);
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < w * h; j++) {
            buffer[j] = field[4 * j + i];
          }
          normalize(buffer);
          hdi::utils::valuesToImage(names[i] + std::to_string(_iteration), buffer, w, h, 1);
        }
      }
    }
    #endif // FIELD_IMAGE_OUTPUT

    _iteration++;
  }
}