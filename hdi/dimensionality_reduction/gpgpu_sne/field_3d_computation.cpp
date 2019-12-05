#include "field_computation.h"
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <cstdint>

#define GLSL(version, shader)  "#version " #version "\n" #shader

namespace {
  // Compute shader for 64^3 stencil texture generation
  const char* stencil_src = GLSL(430,
    layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { vec3 Bounds[]; };
    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
    layout(r8, binding = 0) writeonly uniform image3D stencil_texture;

    uniform uint num_points;

    void main() {
      uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
      if (i >= num_points) {
        return;
      }

      // Get the point position in texture
      vec3 range = Bounds[1] - Bounds[0];
      vec3 pos = imageSize(stencil_texture) * (Positions[i] - Bounds[0]) / range;
      ivec3 minp = ivec3(floor(pos));
      ivec3 maxp = ivec3(ceil(pos));

      // fill all nearest neighboring pixels (not exactly clever)
      imageStore(stencil_texture, ivec3(minp.x, minp.y, minp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(minp.x, minp.y, maxp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(minp.x, maxp.y, minp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(minp.x, maxp.y, maxp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(maxp.x, minp.y, minp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(maxp.x, minp.y, maxp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(maxp.x, maxp.y, minp.z), uvec4(1));
      imageStore(stencil_texture, ivec3(maxp.x, maxp.y, maxp.z), uvec4(1));
    }
  );

  // Compute shader for 3d fields texture
  // r = density, g = x gradient, b = y gradient, a = z gradient
  const char* fields_src = GLSL(430,
    layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
    layout(std430, binding = 1) buffer BoundsInterface { vec3 Bounds[]; };
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(rgba32f, binding = 0) writeonly uniform image3D fields_texture;
    layout(r8, binding = 1) readonly uniform image3D stencil_texture;

    uniform uint num_points;
    uniform uvec3 texture_size;
    uniform float function_support;

    const uint groupSize = gl_WorkGroupSize.x;
    const uint halfGroupSize = groupSize / 2;
    shared vec4 reductionArray[halfGroupSize];

    void main() {
      uint x = gl_WorkGroupID.x;
      uint y = gl_WorkGroupID.y;
      uint z = gl_WorkGroupID.z;

      // Test stencil texture or ignore pixel
      if (imageLoad(stencil_texture, ivec3(x, y, z)).x == 0) {
        return;
      }

      // Invocation location
      uint lid = gl_LocalInvocationIndex.x;

      // Compute pixel position in domain space
      vec3 min_bounds = Bounds[0];
      vec3 range = Bounds[1] - min_bounds;
      vec3 pos = ((vec3(x, y, z) + vec3(0.5)) / texture_size) * range + min_bounds;
      
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        // Compute S = t_stud, V = t_stud_2
        vec3 t = pos - Positions[i];
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
        imageStore(fields_texture, ivec3(x, y, z), reducedArray);
      }
    }
  );
}

void Compute3DFieldComputation::initialize(unsigned num_points) {
  try {
    _stencil_program.create();
    _stencil_program.addShader(COMPUTE, stencil_src);
    _stencil_program.build();
    _fields_program.create();
    _fields_program.addShader(COMPUTE, fields_src);
    _fields_program.build();

    _stencil_program.bind();
    _stencil_program.uniform1ui("num_points", num_points);
    _fields_program.bind();
    _fields_program.uniform1ui("num_points", num_points);
  } catch (const ShaderLoadingException& e) {
    std::cout << e.what() << std::endl;
  }

  // Generate 3d stencil texture
  glGenTextures(1, &_stencil_texture);
  glBindTexture(GL_TEXTURE_3D, _stencil_texture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  // Generate 3d fields texture
  glGenTextures(1, &_field_texture);
  glBindTexture(GL_TEXTURE_3D, _field_texture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

void Compute3DFieldComputation::clean() {
  glDeleteTextures(2, std::array<GLuint, 2>{_stencil_texture, _field_texture}.data());
  _stencil_program.destroy();
  _fields_program.destroy();
}

void Compute3DFieldComputation::compute(unsigned w, unsigned h, unsigned d, float function_support, unsigned n, GLuint position_buff, GLuint bounds_buff, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z) {
  // Only update texture dimensions if they have changed
  bool fresh = true;
  if (_w != w || _h != h || _d != d) {
    fresh = false;
    _w = w;
    _h = h;
    _d = d;
  }

  // Compute 3d stencil texture
  _stencil_program.bind();
  {
    // Define stencil texture format
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _stencil_texture);
    if (!fresh) {
      glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, w, h, d, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }

    // Bind textures and buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
    glBindImageTexture(0, _stencil_texture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    // Perform computation over groups of points
    glDispatchCompute((n / 64) + 1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
  }
  _stencil_program.release();

  // Compute 3d fields texture
  _fields_program.bind();
  {
    // Define fields texture format
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _field_texture);
    if (!fresh) {
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, w, h, d, 0, GL_RGBA, GL_FLOAT, nullptr);
      _fields_program.uniform3ui("texture_size", w, h, d);
    }

    // Bind textures and buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_buff);
    glBindImageTexture(0, _field_texture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(1, _stencil_texture, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R8);

    // Set uniforms
    _fields_program.uniform1f("function_support", function_support);

    // Perform computation over every pixel
    glDispatchCompute(w, h, d);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
  }
  _fields_program.release();  
}