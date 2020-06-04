#include "hdi/debug/renderer/embedding.hpp"
#include "hdi/debug/utils/window.hpp"
#include <glm/glm.hpp>
#include <imgui.h>
#include <iostream>

#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

GLSL(vert, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout (location = 0) in vec4 vertex;
  layout (location = 0) out vec3 pos;
  layout (location = 0) uniform mat4 uTransform;
  layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

  void main() {
    pos = (vertex.xyz - bounds.min) * bounds.invRange;
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(frag, 450,
  layout (location = 0) in vec3 pos;
  layout (location = 0) out vec4 color;
  layout (location = 1) uniform float uOpacity;

  void main() {
    color = vec4(0.0 + 1.0 * normalize(pos), uOpacity);
  }
);

namespace hdi::dbg {
  EmbeddingRenderer::EmbeddingRenderer()
  : RenderComponent(10, false),
    _isInit(false)
  { }

  EmbeddingRenderer::~EmbeddingRenderer()
  {
    if (_isInit) {
      destr();
    }
  }

  void EmbeddingRenderer::init(size_t n,
                               GLuint embeddingBuffer,
                               GLuint boundsBuffer)
  {
    RenderComponent::init();    

    // Specify shader program
    try {
      _program.create();
      _program.addShader(VERTEX, vert);
      _program.addShader(FRAGMENT, frag);
      _program.build();
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Specify vertex array object
    glCreateVertexArrays(1, &_vertexArray);
    glVertexArrayVertexBuffer(_vertexArray, 0, embeddingBuffer, 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_vertexArray, 0);
    glVertexArrayAttribFormat(_vertexArray, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArray, 0, 0);

    _n = n;
    _boundsBuffer = boundsBuffer;
    _embeddingBuffer = embeddingBuffer;
    _isInit = true;
  }

  void EmbeddingRenderer::destr()
  {
    _isInit = false;
    _program.destroy();
    glDeleteVertexArrays(1, &_vertexArray);
    RenderComponent::destr();
  }

  void EmbeddingRenderer::render(glm::mat4 transform, glm::ivec4 viewport)
  {
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
}