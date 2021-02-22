#pragma once

#include <imgui.h>
#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"

namespace hdi::dbg {
  template <unsigned D>
  class FieldRenderer : private RenderComponent {
  public:
    FieldRenderer();
    ~FieldRenderer();

    void init(GLuint *fieldTexture);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    GLuint *_fieldTexture; // ouch
    GLuint _outputTexture;
    GLuint _sampler;
    ShaderProgram _program;
    
    // ImGui default settings
    ImVec2 _dims = ImVec2(0, 0);
    float _depthValue = 0.5f;
  };
} // hdi::dbg