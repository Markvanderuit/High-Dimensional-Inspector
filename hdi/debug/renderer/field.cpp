#include <iostream>
#include "hdi/debug/renderer/field.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

namespace _2d {
  GLSL(convert_src, 450, 
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
    layout(binding = 0) uniform sampler2D inputSampler;
    layout(binding = 0, rgba32f) restrict writeonly uniform image2D outputImage;
    layout(location = 0) uniform uvec2 outputSize;

    void main() {
      // Check that invocation is inside input texture bounds
      const uvec2 i = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
      if (min(i, outputSize - 1) != i) {
        return;
      }

      // Sample input texture
      vec4 field = texture(inputSampler, (vec2(i) + 0.5) / vec2(outputSize));

      // Write to output texture
      imageStore(outputImage, ivec2(i), vec4(normalize(field.yz), 0, 1));
    }
  );
} // _2d

namespace _3d {
  /* GLSL(max_min_src, 450,
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
    layout(binding = 0) uniform sampler3D inputSampler;
    layout(binding = 0, std430) restrict coherent buffer MinMaxV {
      float minv[4];
      float maxv[4];
    };
    layout(location = 0) uniform uvec2 outputSize;
    layout(location = 1) uniform float depthValue;

    shared vec4 mins[16 * 16];
    shared vec4 maxs[16 * 16];

    void main() {
      // Check that invocation is inside input texture bounds
      const uvec2 i = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
      if (min(i, outputSize - 1) != i) {
        return;
      }


    }
  ); */

  GLSL(convert_src, 450, 
    layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
    layout(binding = 0) uniform sampler3D inputSampler;
    layout(binding = 0, rgba32f) restrict writeonly uniform image2D outputImage;
    layout(location = 0) uniform uvec2 outputSize;
    layout(location = 1) uniform float depthValue;

    void main() {
      // Check that invocation is inside input texture bounds
      const uvec2 i = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
      if (min(i, outputSize - 1) != i) {
        return;
      }

      // Sample input texture
      vec4 field = texture(inputSampler, vec3((vec2(i) + 0.5) / vec2(outputSize), depthValue));

      // Write to output texture
      vec3 v = vec3(0.5f);
      if (field != vec4(0)) {
        v += 0.5 * normalize(field.yzw);
      }
      imageStore(outputImage, ivec2(i), vec4(v, 1));
    }
  );
} // _3d


namespace hdi::dbg {
  template <unsigned D>
  FieldRenderer<D>::FieldRenderer()
  : RenderComponent(2, false),
    _fieldTexture(nullptr) {}

  template <unsigned D>
  FieldRenderer<D>::~FieldRenderer() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void FieldRenderer<D>::init(GLuint *fieldTexture) {
    RenderComponent::init();
    if (!_isInit) {
      return;
    }
    _fieldTexture = fieldTexture;
    _dims = ImVec2(0, 0);
    glCreateTextures(GL_TEXTURE_2D, 1, &_outputTexture);
    
    // Build shader program
    try {
      _program.create();
      if constexpr (D == 2) {
        _program.addShader(COMPUTE, _2d::convert_src);
      } else if constexpr (D == 3) {
        _program.addShader(COMPUTE, _3d::convert_src);
      }
      _program.build();
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }
  }

  template <unsigned D>
  void FieldRenderer<D>::destr() {
    if (!_isInit) {
      return;
    }
    _fieldTexture = nullptr;
    _dims = ImVec2(0, 0);
    glDeleteTextures(1, &_outputTexture);
    RenderComponent::destr();
  }

  template <unsigned D>
  void FieldRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport) {
    ASSERT_GL("FieldRenderer::render::begin()");

    if (!_isInit || !_fieldTexture || !glIsTexture(*_fieldTexture)) {
      return;
    }

    ImGui::Begin("Field Rendering");
    ImVec2 dims = ImGui::GetWindowSize();
    if constexpr (D == 3) {
      ImGui::SliderFloat("Depth Value", &_depthValue, 0.f, 1.f);
    }

    // Allocate right sized output texture
    if (_dims.x != dims.x || _dims.y != dims.y) {
      _dims = dims;
      // Re-initialize texture
      glDeleteTextures(1, &_outputTexture);
      glCreateTextures(GL_TEXTURE_2D, 1, &_outputTexture);
      glTextureStorage2D(_outputTexture, 1, GL_RGBA32F, _dims.x, _dims.y);
    }

    // Resample input texture
    {
      _program.bind();
      glBindTextureUnit(0, *_fieldTexture);
      glBindImageTexture(0, _outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      _program.uniform2ui("outputSize", _dims.x, _dims.y);
      if constexpr (D == 3) {
        _program.uniform1f("depthValue", _depthValue);
      }
      glDispatchCompute(dr::ceilDiv(static_cast<unsigned>(_dims.x), 16u), 
                        dr::ceilDiv(static_cast<unsigned>(_dims.y), 16u), 
                        1);
      glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    }

    ImGui::Image((void *) (intptr_t) _outputTexture, ImVec2(_dims.x, _dims.y));
    ImGui::End();

    ASSERT_GL("FieldRenderer::render::end()");
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class FieldRenderer<2>;
  template class FieldRenderer<3>;
} // hdi::dbg