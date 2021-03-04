/*
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <imgui.h>
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/debug/renderer/dual_hierarchy.hpp"
#include "hdi/debug/renderer/dual_hierarchy_shaders_2d.h"

namespace {
  std::array<glm::vec2, 2> lineVertices = {
    glm::vec2(0), glm::vec2(1, 0)
  };

  std::array<unsigned, 2> lineIndices = {
    0, 1
  };

  // Double quads per instance
  std::array<glm::vec2, 16> quadVertices = {
    glm::vec2(-0.5, -0.5), glm::vec2(0.5, -0.5), 
    glm::vec2(0.5, -0.5), glm::vec2(0.5, 0.5), 
    glm::vec2(0.5, 0.5), glm::vec2(-0.5, 0.5),
    glm::vec2(-0.5, 0.5), glm::vec2(-0.5, -0.5),
    glm::vec2(-0.5, -0.5), glm::vec2(0.5, -0.5), 
    glm::vec2(0.5, -0.5), glm::vec2(0.5, 0.5), 
    glm::vec2(0.5, 0.5), glm::vec2(-0.5, 0.5),
    glm::vec2(-0.5, 0.5), glm::vec2(-0.5, -0.5)
  };
}

namespace hdi::dbg {
  template <unsigned D>
  DualHierarchyRenderer<D>::DualHierarchyRenderer()
  : RenderComponent(9, false)
  { }

  template <unsigned D>
  DualHierarchyRenderer<D>::~DualHierarchyRenderer() {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void DualHierarchyRenderer<D>::init(const dr::FieldHierarchy<D> &fieldHierarchy,
                                      const dr::EmbeddingBVH<D> &embeddingHierarchy,
                                      GLuint boundsBuffer,
                                      GLuint debugQueueBuffer,
                                      GLuint debugHeadBuffer) {
    RenderComponent::init();
    if (!_isInit) {
      return;
    }

    // This is gonna bite me
    _boundsBuffer = boundsBuffer;
    _debugQueueBuffer = debugQueueBuffer;
    _debugHeadBuffer = debugHeadBuffer;
    _fieldHierarchyPtr = &fieldHierarchy;
    _embeddingHierarchyPtr = &embeddingHierarchy;

    // Create opengl objects
    glCreateBuffers(_buffers.size(), _buffers.data());
    glCreateVertexArrays(_vertexArrays.size(), _vertexArrays.data());

    // Specify shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }

      if constexpr (D == 2) {
        _programs(ProgramType::eLines).addShader(VERTEX, _2d::lines_vert_src);
        _programs(ProgramType::eLines).addShader(FRAGMENT, _2d::lines_frag_src);
        _programs(ProgramType::eQuads).addShader(VERTEX, _2d::nodes_vert_src);
        _programs(ProgramType::eQuads).addShader(FRAGMENT, _2d::nodes_frag_src);
      } else if constexpr (D == 3) {
        // ...
      }

      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException &e) {
      std::cerr << e.what() << std::endl;
      std::exit(0); // yeah no, not recovering from this catastrophy
    }

    // Specify buffer objects
    {
      glNamedBufferStorage(_buffers(BufferType::eQuadVertices), quadVertices.size() * sizeof(glm::vec2), quadVertices.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eLineVertices), lineVertices.size() * sizeof(glm::vec2), lineVertices.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eLineIndices), lineIndices.size() * sizeof(unsigned), lineIndices.data(), 0);
    }
    
    // Specify vertex array objects
    {
      // VAO for line data, instanced with node pairs
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eLine), _buffers(BufferType::eLineIndices));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eLine), 0, _buffers(BufferType::eLineVertices), 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eLine), 1, debugQueueBuffer, 0, sizeof(glm::uvec2));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eLine), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eLine), 1, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eLine), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eLine), 1);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eLine), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eLine), 1, 2, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eLine), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eLine), 1, 1);

      // VAO for quad data, instanced with node pairs
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 0, _buffers(BufferType::eQuadVertices), 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 1, debugQueueBuffer, 0, sizeof(glm::uvec2));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 1);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eQuad), 1, 2, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 1, 1);
    }

    glAssert("dbg::dual_hierarchy::init()");
  }

  template <unsigned D>
  void DualHierarchyRenderer<D>::destr() {
    if (!_isInit) {
      return;
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteVertexArrays(_vertexArrays.size(), _vertexArrays.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _fieldHierarchyPtr = nullptr;
    _embeddingHierarchyPtr = nullptr;
    RenderComponent::destr();
  }

  template <unsigned D>
  void DualHierarchyRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport) {
    if (!_isInit || !_fieldHierarchyPtr || !_embeddingHierarchyPtr) {
      return;
    }

    ImGui::Begin("Dualhierarchy Rendering");
    ImGui::Checkbox("Draw", &_isActive);
    if (!_isActive) {
      ImGui::End();
      return;
    }

    // Level/mouse slider
    const uint lvlMin = 0u;
    const uint lvlMax = _fieldHierarchyPtr->layout().nLvls - 1u;
    ImGui::SliderScalar("Level", ImGuiDataType_U32, &_renderLvl, &lvlMin, &lvlMax);
    ImGui::SliderFloat("mouse x", &mousex, 0.f, 1.f);
    ImGui::SliderFloat("mouse y", &mousey, 0.f, 1.f);

    // Dirty, but find nr. of pairs to draw
    uint head;
    glGetNamedBufferSubData(_debugHeadBuffer, 0, sizeof(uint), &head);

    // Draw lines
    {
      auto &program = _programs(ProgramType::eLines);
      program.bind();
      program.uniformMatrix4f("transform", &transform[0][0]);
      program.uniform1ui("lvl", _renderLvl);
      program.uniform2f("mousepos", mousex, mousey);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _fieldHierarchyPtr->buffers().field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _embeddingHierarchyPtr->buffers().node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _embeddingHierarchyPtr->buffers().node1);

      // Draw instanced line for each pair
      glLineWidth(1.f);
      glBindVertexArray(_vertexArrays(VertexArrayType::eLine));
      glDrawElementsInstanced(GL_LINES, 2, GL_UNSIGNED_INT, nullptr, head);
    }

    // Draw quads
    {
      auto &program = _programs(ProgramType::eQuads);
      program.bind();
      program.uniformMatrix4f("transform", &transform[0][0]);
      program.uniform1ui("lvl", _renderLvl);
      program.uniform2f("mousepos", mousex, mousey);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _fieldHierarchyPtr->buffers().field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _embeddingHierarchyPtr->buffers().node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _embeddingHierarchyPtr->buffers().node1);

      // Draw instanced pair of quads for each pair
      glLineWidth(2.f);
      glBindVertexArray(_vertexArrays(VertexArrayType::eQuad));
      glDrawArraysInstanced(GL_LINES, 0, 16, head);
    }

    ImGui::End();
    glAssert("DualHierarchyRenderer::render::end()");
  }

  // Explicit template instantiations for 2 and 3 dimensions
  template class DualHierarchyRenderer<2>;
  template class DualHierarchyRenderer<3>;
} // namespace hdi::db
