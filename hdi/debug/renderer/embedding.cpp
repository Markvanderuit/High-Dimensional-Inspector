#include "hdi/debug/renderer/embedding.hpp"
#include "hdi/debug/utils/window.hpp"
#include <imgui.h>
#include <iostream>

#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

namespace _2d {
  GLSL(vert, 450,
    struct Bounds {
      vec2 min;
      vec2 max;
      vec2 range;
      vec2 invRange;
    };

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(location = 0) in vec2 vertex;
    layout(location = 0) out vec2 pos;
    layout(location = 0) uniform mat4 uTransform;

    void main() {
      pos = (vertex.xy - bounds.min) * bounds.invRange;
      gl_Position = uTransform * vec4(pos, 0, 1);
    }
  );

  GLSL(frag, 450,
    layout(location = 0) in vec2 pos;
    layout(location = 0) out vec4 color;
    layout(location = 1) uniform float uOpacity;

    void main() {
      vec2 _pos = normalize(pos);
      color = vec4(0.f, _pos.x, _pos.y, uOpacity);
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

    layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
    layout(location = 0) in vec4 vertex;
    layout(location = 0) out vec3 pos;
    layout(location = 0) uniform mat4 uTransform;

    void main() {
      pos = (vertex.xyz - bounds.min) * bounds.invRange;
      gl_Position = uTransform * vec4(pos, 1);
    }
  );

  GLSL(frag, 450,
    layout(location = 0) in vec3 pos;
    layout(location = 0) out vec4 color;
    layout(location = 1) uniform float uOpacity;

    void main() {
      color = vec4(0.0 + 1.0 * normalize(pos), uOpacity);
    }
  );
} // namespace _3d

namespace hdi::dbg {
  template <unsigned D>
  EmbeddingRenderer<D>::EmbeddingRenderer()
  : RenderComponent(10, false) { }

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
    ImGui::SliderFloat("Point size", &_pointSize, 0.5f, 16.f, "%1.1f");
    ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.05f, 1.f, "%.2f");
    ImGui::End();
    
    if (_draw) {
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindVertexArray(_vertexArray);
      glPointSize(_pointSize);
      _program.bind();
      _program.uniformMatrix4f("uTransform", &transform[0][0]);
      _program.uniform1f("uOpacity", _pointOpacity);
      glDrawArrays(GL_POINTS, 0, _n);
      _program.release();
    }
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class EmbeddingRenderer<2>;
  template class EmbeddingRenderer<3>;
}