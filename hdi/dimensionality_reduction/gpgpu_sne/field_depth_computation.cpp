#include "field_depth_computation.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include <algorithm>

#define GLSL(version, shader)  "#version " #version "\n" #shader

namespace {
  void normalize(std::vector<float>& b) {
    const float max = *std::max_element(b.begin(), b.end());
    const float min = *std::min_element(b.begin(), b.end());
    const float range = max - min;
    std::transform(b.begin(), b.end(), b.begin(),
      [&range, &min](const auto& v) { return (v - min) / range; });
  }

  std::vector<float> gradientMap(const std::vector<float>& b) {
    std::vector<float> _b(b);
    normalize(_b);

    std::vector<float> g(_b.size() * 3, 0.f);
    for (int i = 0; i < _b.size(); i++) {
      // if (_b[i] > 0.f) {
        g[i * 3 + 0] = 1.f - _b[i];
        g[i * 3 + 2] = _b[i];
      // }
    }

    return g;
  }

  const char* stencil_vert_src = GLSL(430,
    layout (location = 0) in vec3 point;
    
    uniform vec3 min_bounds;
    uniform vec3 range;

    void main() {
      gl_Position = vec4(((point - min_bounds) / range) * 2 - 1, 1);
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
    layout (std430, binding = 0) buffer PosInterface { vec3 Positions[]; };
    layout (std430, binding = 1) buffer BoundsInterface { vec3 Bounds[]; };
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(rg32f, binding = 0) readonly uniform image2D stencil_depth_texture;
    layout(rgba32f, binding = 1) writeonly uniform image2D fields_texture;

    uniform uint num_points;
    uniform uvec2 texture_size;
    uniform float function_support;

    // Reduction components
    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec4 reductionArray[halfGroupSize];

    void main() {
      // Pixels of current workgroup
      uint x = gl_WorkGroupID.x;
      uint y = gl_WorkGroupID.y;

      // ID of current local instance
      uint lid = gl_LocalInvocationIndex.x;

      // Test stencil texture or ignore pixel
      vec2 stencil_depth = imageLoad(stencil_depth_texture, ivec2(x, y)).xy;
      if (stencil_depth.x == 0.f) {
        // TODO: Remove this store, is just for debugging
        if (lid == 0)
          imageStore(fields_texture, ivec2(x, y), vec4(0));
        return;
      }

      // Obtain bounds
      vec3 min_bounds = Bounds[0];
      vec3 max_bounds = Bounds[1];
      vec3 range = max_bounds - min_bounds;

      // Compute pixel position in the embedding domain (x, y only)
      vec2 pixel_pos = (vec2(x, y) + vec2(0.5)) / texture_size * range.xy + min_bounds.xy;
      
      // Domain pos uses closest measured depth for determining z
      vec3 domain_pos = vec3(pixel_pos, stencil_depth.y * range.z + min_bounds.z);

      // Iterate over points to obtain density/gradient
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        vec3 t = domain_pos - Positions[i];
        float eucl_2 = dot(t, t);
        float t_stud = 1.f / (1.f + eucl_2);
        float t_stud_2 = t_stud * t_stud;

        // 3d field layout is: S, V.x, V.y, V.z
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


  void glAssert(const std::string& msg) {
    GLenum err; 
    while ((err = glGetError()) != GL_NO_ERROR) {
      std::cerr << "Location: " << msg << ", err: " << err << std::endl;
      exit(0);
    }
  }
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
      std::cerr << "Bleep " << e.what() << std::endl;
      exit(0); // yeah no, not continuing after that
    }

    // Generate 2d stencil + depth texture
    glGenTextures(1, &_textures[STENCIL_DEPTH]);
    glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    glAssert("depth field depth buffer");

    // Generate framebuffer for stencil/depth pass
    glGenFramebuffers(1, &_stencil_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _stencil_fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textures[STENCIL_DEPTH], 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depth_buffer);
    glAssert("depth field framebuffer");
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenVertexArrays(1, &_point_vao);

    glAssert("depth field init");
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

  void DepthFieldComputation::compute(unsigned w, unsigned h, unsigned d,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff,
                float min_x, float min_y, float min_z,
                float max_x, float max_y, float max_z) {
    // Compute stencil and texture
    { 
      // Configure framebuffer texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, w, h, 0, GL_RG, GL_FLOAT, nullptr);

      // Configure renderbuffer
      glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);

      // Configure uniforms
      _stencil_program.bind();
      _stencil_program.uniform3f("min_bounds", min_x, min_y, min_z);
      _stencil_program.uniform3f("range", max_x - min_x, max_y - min_y, max_z - min_z);
      
      // Prepare for drawing, we need a depth test
      glViewport(0, 0, w, h);
      glBindFramebuffer(GL_FRAMEBUFFER, _stencil_fbo);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glDepthMask(GL_TRUE);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
      // Draw n points with 3px sized border
      glBindVertexArray(_point_vao);
      glBindBuffer(GL_ARRAY_BUFFER, position_buff);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glPointSize(3);
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

    // Test read stencil/depth texture
    // #define field_image_output
    #ifdef field_image_output
    if (_iteration % 100 == 0) {
    // if (_iteration == 999) {
      // Output depth/stencil
      {
        std::vector<float> stencil_depth(w * h * 2, 0.f);
        glBindTexture(GL_TEXTURE_2D, _textures[STENCIL_DEPTH]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, stencil_depth.data());

        const std::array<std::string, 2> names_0 = { 
          "images/stencil_",
          "images/depth_" 
        };

        std::vector<float> buffer(w * h, 0.f);
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < w * h; j++) {
            buffer[j] = stencil_depth[2 * j + i];
          }
          // normalize(buffer);
          hdi::utils::valuesToImage(names_0[i] + std::to_string(_iteration), buffer, w, h, 1);
        }
      }

      // Output fields
      {
        std::vector<float> field(w * h * 4, 0.f);
        glBindTexture(GL_TEXTURE_2D, _textures[FIELD]);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, field.data());
      
        const std::array<std::string, 4> names_1 = { 
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
          hdi::utils::valuesToImage(names_1[i] + std::to_string(_iteration), buffer, w, h, 1);
        }
      }
    }
    #endif // field_image_output

    _iteration++;
  }
}