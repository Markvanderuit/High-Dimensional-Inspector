#include "field_depth_computation.h"
#include "hdi/visualization/opengl_helpers.h"
#include "hdi/utils/visual_utils.h"
#include <algorithm>
#include <QMatrix4x4>
#include <string>

#define GLSL(name, version, shader) \
  static const char* name = \
  "#version " #version "\n" #shader
// #define FIELD_IMAGE_OUTPUT // Output field images every 100 iterations

namespace {
  // Magic numbers
  constexpr float pointSize = 1.f;

  // For debug output
  void normalize(std::vector<float>& b) {
    const float max = *std::max_element(b.begin(), b.end());
    const float min = *std::min_element(b.begin(), b.end());
    const float range = max - min;
    std::transform(b.begin(), b.end(), b.begin(),
      [&range, &min](const auto& v) { return (v - min) / range; });
  }

  GLSL(depth_vert_src, 430,
    layout(location = 0) in vec4 point;
    layout(std430, binding = 0) buffer BoundsInterface { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    uniform mat4 transformation;

    void main() {
      // Transform point into [0, 1] space
      vec3 position = (point.xyz - minBounds) * invRange;

      // Apply camera transformation
      position = vec3(transformation * vec4(position, 1));
      
      // Transform point into [-1, 1] clip space
      gl_Position =  vec4(position * 2.f - 1.f, 1.f);
    }
  );

  GLSL(depth_geom_src, 430,
    layout (points) in;
    layout (points, max_vertices = 1) out;

    void main() {
      // Emit point for front depth boundary
      gl_Layer = 0;
      gl_Position = gl_in[0].gl_Position;
      EmitVertex();
      EndPrimitive();

      // Emit point for back depth boundary
      gl_Layer = 1;
      gl_Position = gl_in[0].gl_Position * vec4(1, 1, -1, 1);
      EmitVertex();
      EndPrimitive();
    }
  );

  GLSL(depth_frag_src, 430,
    void main() {
      gl_FragDepth = 0;
    }
  );

  GLSL(field_3D_src, 430,
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    layout(std430, binding = 0) buffer PosInterface { vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { 
      vec3 minBounds;
      vec3 maxBounds;
      vec3 range;
      vec3 invRange;
    };
    layout(rgba32f, binding = 0) writeonly uniform image3D fields_texture;
    
    uniform uint num_points;
    uniform uvec3 texture_size;
    uniform mat4 invTransformation;
    uniform sampler2DArray depth_textures;

    // Reduction components
    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec4 reductionArray[halfGroupSize];

    void main() {
      // Pixel location of current workgroup
      ivec3 xyz = ivec3(gl_WorkGroupID.xyz);

      // ID of current local instance
      uint lid = gl_LocalInvocationIndex.x;
      
      // Query depth map
      float zNear = texelFetch(depth_textures, ivec3(xyz.xy, 0), 0).x;
      float zFar = 1.f - texelFetch(depth_textures, ivec3(xyz.xy, 1), 0).x;
      
      // Test depth as stencil
      if ((zNear == 1.f && zFar == 0.f)) {
        if (lid == 0) {
          imageStore(fields_texture, xyz, vec4(0));
        }
        return;
      }
      
      // Map xyz to [0, 1], actually bound by [zNear, zFar] though
      vec3 domain_pos = (vec3(xyz) + vec3(0.5)) / texture_size;

      // Map z from [zNear, zFar] to [0, 1] using queried depth
      domain_pos.z = domain_pos.z * (zFar - zNear) + zNear;

      // Undo camera transformation
      domain_pos = vec3(invTransformation * vec4(domain_pos, 1));

      // Map to domain bounds
      domain_pos = domain_pos * range + minBounds;

      // Iterate over points to obtain density/gradient
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        vec3 t = domain_pos - Positions[i];
        float t_stud = 1.f / (1.f + dot(t, t));
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
        imageStore(fields_texture, xyz, reducedArray);
      }
    }
  );
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

  void DepthFieldComputation::initialize(unsigned n) {
    // Build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_DEPTH].addShader(VERTEX, depth_vert_src);
      _programs[PROGRAM_DEPTH].addShader(GEOMETRY, depth_geom_src);
      // _programs[PROGRAM_DEPTH].addShader(FRAGMENT, depth_frag_src);
      _programs[PROGRAM_FIELD_3D].addShader(COMPUTE, field_3D_src); 
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Generate 2d depth texture array
    glGenTextures(1, &_textures[TEXTURE_DEPTH]);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH]);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate 3d fields texture
    glGenTextures(1, &_textures[TEXTURE_FIELD_3D]);
    glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Generate framebuffer with depth texture attached only
    glGenFramebuffers(1, &_depth_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _depth_fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _textures[TEXTURE_DEPTH], 0);
    glDrawBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    glGenVertexArrays(1, &_point_vao);
    _iteration = 0;
    _initialized = true;
  }

  void DepthFieldComputation::clean() {
    glDeleteTextures(_textures.size(), _textures.data());
    glDeleteFramebuffers(1, &_depth_fbo);
    glDeleteVertexArrays(1, &_point_vao);
    for (auto& program : _programs) {
      program.destroy();
    }
    _iteration = 0;
    _initialized = false;
  }

  void DepthFieldComputation::compute(unsigned w, unsigned h, unsigned d,
                float function_support, unsigned n,
                GLuint position_buff, GLuint bounds_buff,
                Bounds3D bounds, QMatrix4x4 view) {
    QMatrix4x4 inverse = view.inverted();
    Point3D minBounds = bounds.min;
    Point3D range = bounds.range();

    // Only update texture dimensions if they have changed
    bool refresh = false;
    if (_w != w || _h != h || _d != d) {
      refresh = true;
      _w = w;
      _h = h;
      _d = d;
    }

    // Compute depth textures
    { 
      // Configure framebuffer depth texture
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH]);
      if (refresh) {
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, _w, _h, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
      }

      // Bind bounds buffer
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bounds_buff);
      
      // Bind uniforms to shader
      auto& program = _programs[PROGRAM_DEPTH];
      program.bind();
      program.uniformMatrix4f("transformation", view.data());
      
      // Prepare for drawing with depth test
      glViewport(0, 0, _w, _h);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClearDepthf(1.f);
      glBindFramebuffer(GL_FRAMEBUFFER, _depth_fbo);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glDepthMask(GL_TRUE);
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
    }

    // Compute fields 3D texture
    {
      auto &program = _programs[PROGRAM_FIELD_3D];
      program.bind();

      // Define fields texture format
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
      if (refresh) {
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _w, _h, _d, 0, GL_RGBA, GL_FLOAT, nullptr);
      }

      // Attach depth texture for sampling
      glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH]);

      // Set uniforms in compute shader
      program.uniform1ui("num_points", n);
      program.uniform3ui("texture_size", _w, _h, _d);
      program.uniformMatrix4f("invTransformation", inverse.data());
      program.uniform1i("depth_textures", 0);

      // Bind textures and buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
      glBindImageTexture(0, _textures[TEXTURE_FIELD_3D], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Perform computation
      glDispatchCompute(_w, _h, _d);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      
      // Cleanup
      program.release();
    }

    // Test output textures every 100 iterations
    #ifdef FIELD_IMAGE_OUTPUT
    if (_iteration % 10 == 0) {
      // Output depth textures
      {
        std::vector<float> depth(w * h * 2, 0.f);
        glBindTexture(GL_TEXTURE_2D_ARRAY, _textures[TEXTURE_DEPTH]);
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data());
        const std::string name = "images/depth_";
        
        std::vector<float> buffer(w * h, 0.f);
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < w * h; j++) {
            buffer[j] = depth[i *  w * h + j];
          }
          // normalize(buffer);
          hdi::utils::valuesToImage(name + std::to_string(i) + "_" + std::to_string(_iteration), buffer, w, h, 1);
        }
      }
      
      // Output 3d fields texture
/*       {
        std::vector<float> field(w * h * d * 4, 0.f);
        glBindTexture(GL_TEXTURE_3D, _textures[TEXTURE_FIELD_3D]);
        glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, field.data());

        const std::array<std::string, 4> names = { 
          "images/3D/density_",
          "images/3D/gradient_x_",
          "images/3D/gradient_y_",
          "images/3D/gradient_z_" 
        };

        std::vector<float> buffer(w * h, 0.f);
        for (int k = 0; k < d; k++) {
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < w * h; j++) {
              buffer[j] = field[(k * w * h * 4) + 4 * j + i];
            }
            normalize(buffer);
            hdi::utils::valuesToImage(names[i] + std::to_string(_iteration) + "_" + std::to_string(k), buffer, w, h, 1);
          }
        }
      } */
    }
    #endif // FIELD_IMAGE_OUTPUT

    _iteration++;
  }
}