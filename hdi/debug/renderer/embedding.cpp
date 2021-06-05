#include <array>
#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/embedding.hpp"
#include "hdi/debug/renderer/embedding_shaders_2d.h"
#include "hdi/debug/renderer/embedding_shaders_3d.h"
#include "hdi/debug/utils/window.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

namespace {
  std::array<glm::vec2, 4> quadVertices = {
    glm::vec2(-0.5, -0.5),  // 0
    glm::vec2(0.5, -0.5),   // 1
    glm::vec2(0.5, 0.5),    // 2
    glm::vec2(-0.5, 0.5)    // 3
  };
  
  std::array<glm::vec2, 4> quadUvs = {
    glm::vec2(0.f, 0.f),  // 0
    glm::vec2(1.f, 0.f),  // 1
    glm::vec2(1.f, 1.f),  // 2
    glm::vec2(0.f, 1.f)   // 3
  };

  std::array<unsigned, 6> quadIndices = {
    0, 1, 2,  2, 3, 0
  };
}

namespace hdi::dbg {
  template <unsigned D>
  EmbeddingRenderer<D>::EmbeddingRenderer()
  : RenderComponent(999, false), _hasLabels(false) { }

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
      for (auto& program : _programs) {
        program.create();
      }
      if constexpr (D == 2) {
        _programs(ProgramType::eQuad).addShader(VERTEX, _2d::vert_quad);
        _programs(ProgramType::eQuad).addShader(FRAGMENT, _2d::frag_quad);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eQuad).addShader(VERTEX, _3d::vert_quad);
        _programs(ProgramType::eQuad).addShader(FRAGMENT, _3d::frag_quad);
      }

      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Create opengl objects
    glCreateBuffers(_buffers.size(), _buffers.data());
    glCreateVertexArrays(_vertexArrays.size(), _vertexArrays.data());

    // Specify buffers for quad data
    glNamedBufferStorage(_buffers(BufferType::eQuadVertices), quadVertices.size() * sizeof(glm::vec2), quadVertices.data(), 0);
    glNamedBufferStorage(_buffers(BufferType::eQuadUvs), quadUvs.size() * sizeof(glm::vec2), quadUvs.data(), 0);
    glNamedBufferStorage(_buffers(BufferType::eQuadIndices), quadIndices.size() * sizeof(unsigned), quadIndices.data(), 0);

    // Specify vertex array for quad data
    glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 0, _buffers(BufferType::eQuadVertices), 0, sizeof(glm::vec2));
    glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 1, embeddingBuffer, 0, ((D > 2) ? 4 : 2) * sizeof(float));
    glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eQuad), _buffers(BufferType::eQuadIndices));
    glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 0, 0);
    glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 1, 1);
    glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 0);
    glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 1);
    glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 1, ((D > 2) ? 3 : 2), GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 0, 0);
    glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 1, 1);

    // Add labels to vertex array object if they exist
    const auto &manager = RenderManager::instance();
    if (manager.isInit()) {
      GLuint labelsBuffer = manager.labelsBuffer();
      if (labelsBuffer) {
        _hasLabels = true;
        glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 2, labelsBuffer, 0, sizeof(unsigned));
        glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 2, 1);
        glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 2);
        glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eQuad), 2, 1, GL_UNSIGNED_INT, 0);
        glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 2, 2);
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
    for (auto& program : _programs) {
      program.destroy();
    }
    glDeleteVertexArrays(_vertexArrays.size(), _vertexArrays.data());
    glDeleteBuffers(_buffers.size(), _buffers.data());
    RenderComponent::destr();
  }

  template <unsigned D>
  void EmbeddingRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport) {
    glAssert("EmbeddingRenderer::render::begin()");

    if (!_isInit) {
      return;
    }

#ifdef INSERT_IMGUI
    ImGui::Begin("Embedding Rendering");
    ImGui::Checkbox("Draw", &_draw);
    if (_draw) {
      if (_hasLabels) {
        ImGui::Checkbox("Draw labels", &_drawLabels);
      } else {
        _drawLabels = false; // nope
      }
      ImGui::SliderFloat("Point size", &_quadSize, 0.001f, 0.05f);
      if constexpr (D == 2) {
        ImGui::Checkbox("Draw gaussiuan", &_drawGaussian);
        if (_drawGaussian) {
          ImGui::SliderFloat("Point opacity", &_pointOpacity, 0.05f, 1.f, "%.2f");
        }
      }
    }
    ImGui::End();
#endif // INSERT_IMGUI

    if (_draw) {
      // Set program uniforms
      auto& program = _programs(ProgramType::eQuad);
      program.bind();
      program.uniformMatrix4f("transform", &transform[0][0]);
      program.uniform1f("size", _quadSize);
      program.uniform1ui("drawLabels", _drawLabels ? 1 : 0);
      if constexpr (D == 2) {
        program.uniform1ui("drawGaussian", _drawGaussian ? 1 : 0);
        program.uniform1f("opacity", _drawGaussian ? _pointOpacity : 1.f);
      }

      // Specify buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);

      // Specify VAO bindings
      glBindVertexArray(_vertexArrays(VertexArrayType::eQuad));

      // Depth write is temporarily disabled for 2D, so we can perform blending in any order
      if constexpr (D == 2) {
        glDepthMask(GL_FALSE);
      }

      // Specify OpenGL config and perform draw
      glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, _n);
      
      // Depth write is temporarily disabled for 2D, so we can perform blending in any order
      if constexpr (D == 2) {
        glDepthMask(GL_TRUE);
      }
    }
    glAssert("EmbeddingRenderer::render::end()");
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class EmbeddingRenderer<2>;
  template class EmbeddingRenderer<3>;
}