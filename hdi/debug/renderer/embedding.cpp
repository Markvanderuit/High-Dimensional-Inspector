#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/embedding.hpp"
#include "hdi/debug/utils/window.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

namespace _2d {
  GLSL(vert, 450,
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    const vec3 labels[10] = vec3[10](
      vec3(16, 78, 139),
      vec3(139, 90, 43),
      vec3(138, 43, 226),
      vec3(0, 128, 0),
      vec3(255, 150, 0),
      vec3(204, 40, 40),
      vec3(131, 139, 131),
      vec3(0, 205, 0),
      vec3(20, 20, 20),
      vec3(0, 150, 255)
    );

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(location = 0) in vec2 vertex;
    layout(location = 1) in uint label;
    layout(location = 0) out vec4 color;
    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform bool drawLabels;
    layout(location = 2) uniform float uOpacity;

    void main() {
      vec2 pos = (vertex.xy - bounds.min) * bounds.invRange;
      if (drawLabels) {
        color = vec4(labels[label % 10] / 255.f, uOpacity);
      } else {
        vec2 _pos = normalize(pos);
        color = vec4(_pos.x, abs(_pos.y - _pos.x), _pos.y, uOpacity); // accidental rainbow, neat!
      }
      gl_Position = uTransform * vec4(pos, 0, 1);
    }
  );

  GLSL(frag, 450,
    layout(location = 0) in vec4 inColor;
    layout(location = 0) out vec4 outColor;

    void main() {
      outColor = inColor;
    }
  );
} // namespace _2d

namespace _3d {
  GLSL(vert, 450,
    struct Bounds {
      vec3 min;
      vec3 max;
      vec3 range;
      vec3 invRange;
    };

    const vec3 labels[10] = vec3[10](
      vec3(16, 78, 139),
      vec3(139, 90, 43),
      vec3(138, 43, 226),
      vec3(0, 128, 0),
      vec3(255, 150, 0),
      vec3(204, 40, 40),
      vec3(131, 139, 131),
      vec3(0, 205, 0),
      vec3(20, 20, 20),
      vec3(0, 150, 255)
    );

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(location = 0) in vec4 vertex;
    layout(location = 1) in uint label;
    layout(location = 0) out vec4 color;
    layout(location = 0) uniform mat4 uTransform;
    layout(location = 1) uniform bool drawLabels;
    layout(location = 2) uniform float uOpacity;

    void main() {
      vec3 pos = (vertex.xyz - bounds.min) * bounds.invRange;
      if (drawLabels) {
        color = vec4(labels[label % 10] / 255.f, uOpacity);
      } else {
        color = vec4(normalize(pos), uOpacity);
      }
      gl_Position = uTransform * vec4(pos, 1);
    }
  );

  GLSL(frag, 450,
    layout(location = 0) in vec4 inColor;
    layout(location = 0) out vec4 outColor;

    void main() {
      outColor = inColor;
    }
  );
} // namespace _3d

namespace hdi::dbg {
  template <unsigned D>
  EmbeddingRenderer<D>::EmbeddingRenderer()
  : RenderComponent(10, false), _hasLabels(false) { }

  template <unsigned D>
  EmbeddingRenderer<D>::~EmbeddingRenderer() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void EmbeddingRenderer<D>::init(size_t n,
                                  GLuint embeddingBuffer,
                                  GLuint boundsBuffer) {
    RenderComponent::init();    
    if (!_isInit) {
      return;
    }

    // Specify shader program
    try {
      _program.create();
      if constexpr (D == 2) {
        _program.addShader(VERTEX, _2d::vert);
        _program.addShader(FRAGMENT, _2d::frag);
      } else if constexpr (D == 3) {
        _program.addShader(VERTEX, _3d::vert);
        _program.addShader(FRAGMENT, _3d::frag);
      }
      _program.build();
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Specify vertex array object
    glCreateVertexArrays(1, &_vertexArray);
    glVertexArrayVertexBuffer(_vertexArray, 0, embeddingBuffer, 0, ((D > 2) ? 4 : 2) * sizeof(float));
    glEnableVertexArrayAttrib(_vertexArray, 0);
    glVertexArrayAttribFormat(_vertexArray, 0, ((D > 2) ? 3 : 2), GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArray, 0, 0);

    // Add labels to vertex array object if they exist
    const auto *ptr = RenderManager::currentManager();
    if (ptr) {
      GLuint labelsBuffer = ptr->labelsBuffer();
      if (labelsBuffer) {
        _hasLabels = true;
        glVertexArrayVertexBuffer(_vertexArray, 1, labelsBuffer, 0, sizeof(unsigned));
        glEnableVertexArrayAttrib(_vertexArray, 1);
        glVertexArrayAttribIFormat(_vertexArray, 1, 1, GL_UNSIGNED_INT, 0);
        glVertexArrayAttribBinding(_vertexArray, 1, 1);
      } else {
        _hasLabels = false;
      }
    }

    _n = n;
    _boundsBuffer = boundsBuffer;
    _embeddingBuffer = embeddingBuffer;
  }

  template <unsigned D>
  void EmbeddingRenderer<D>::destr() {
    if (!_isInit) {
      return;
    }
    _program.destroy();
    glDeleteVertexArrays(1, &_vertexArray);
    RenderComponent::destr();
  }

  template <unsigned D>
  void EmbeddingRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport) {
    if (!_isInit) {
      return;
    }

    ImGui::Begin("Embedding Rendering");
    ImGui::Checkbox("Draw", &_draw);
    if (_draw) {
      if (_hasLabels) {
        ImGui::Checkbox("Draw labels", &_drawLabels);
      } else {
        _drawLabels = false; // nope
      }
      ImGui::SliderFloat("Point size", &_pointSize, 0.5f, 16.f, "%1.1f");
      ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.05f, 1.f, "%.2f");
    }
    ImGui::End();
    
    if (_draw) {
      _program.bind();
      _program.uniformMatrix4f("uTransform", &transform[0][0]);
      _program.uniform1f("uOpacity", _pointOpacity);
      _program.uniform1ui("drawLabels", _drawLabels ? 1 : 0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindVertexArray(_vertexArray);
      glPointSize(_pointSize);
      glDrawArrays(GL_POINTS, 0, _n);
      _program.release();
    }
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class EmbeddingRenderer<2>;
  template class EmbeddingRenderer<3>;
}