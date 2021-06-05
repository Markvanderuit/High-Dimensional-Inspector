#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/field_hierarchy.hpp"
#include "hdi/debug/renderer/field_hierarchy_shaders_2d.h"
#include "hdi/debug/renderer/field_hierarchy_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

GLSL(flags_comp, 450,
  // Wrapper structure for pair data
  struct Pair {
    uint f;       // Field hierarchy node index
    uint e;       // Embedding hierarchy node index
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  layout(binding = 0, std430) restrict readonly buffer Debu { Pair pairsBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer DebH { uint pairsHead; };
  layout(binding = 2, std430) restrict writeonly buffer Fla { uint flagsBuffer[]; };

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (i >= pairsHead) {
      return;
    }

    Pair pair = pairsBuffer[i];
    flagsBuffer[pair.f] = 1;
  }
);

namespace {
  std::array<glm::vec2, 4> quadVertices = {
    glm::vec2(-0.5, -0.5),  // 0
    glm::vec2(0.5, -0.5),   // 1
    glm::vec2(0.5, 0.5),    // 2
    glm::vec2(-0.5, 0.5)    // 3
  };

  std::array<unsigned, 8> quadLineIndices = {
    0, 1,  1, 2,  2, 3,  3, 0
  };

  std::array<unsigned, 6> quadTriangleIndices = {
    0, 1, 2,  2, 3, 0
  };

  std::array<glm::vec4, 8> cubeVertices = {
    glm::vec4(-0.5, -0.5, -0.5, 0.f),  // 0
    glm::vec4(0.5, -0.5, -0.5, 0.f),   // 1
    glm::vec4(0.5, 0.5, -0.5, 0.f),    // 2
    glm::vec4(-0.5, 0.5, -0.5, 0.f),   // 3
    glm::vec4(-0.5, -0.5, 0.5, 0.f),   // 4
    glm::vec4(0.5, -0.5, 0.5, 0.f),    // 5
    glm::vec4(0.5, 0.5, 0.5, 0.f),     // 6
    glm::vec4(-0.5, 0.5, 0.5, 0.f),    // 7
  };

  std::array<unsigned, 24> cubeIndices = {
    0, 1,  1, 2,  2, 3,  3, 0,    // bottom
    0, 4,  1, 5,  2, 6,  3, 7,    // sides
    4, 5,  5, 6,  6, 7,  7, 4     // top
  };
}

namespace hdi::dbg {
  template <unsigned D>
  FieldHierarchyRenderer<D>::FieldHierarchyRenderer()
  : RenderComponent(5, false),
    _hierarchy(nullptr),
    _fieldLvl(1)
  { }

  template <unsigned D>
  FieldHierarchyRenderer<D>::~FieldHierarchyRenderer()
  {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void FieldHierarchyRenderer<D>::init(const dr::FieldHierarchy<D> &hierarchy, GLuint boundsBuffer, GLuint pairsBuffer, GLuint pairsHead)
  {
    RenderComponent::init();
    if (!_isInit) {
      return;
    }

    // This is gonna bite me
    _hierarchy = &hierarchy;
    
    // Query hierarchy for layout specs and buffer handles
    const auto layout = hierarchy.layout();
    const auto buffers = hierarchy.buffers();

    // Create and build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      if constexpr (D == 2) {
        _programs(ProgramType::eFieldSlice).addShader(COMPUTE, _2d::field_slice_src);
        _programs(ProgramType::eBvhDraw).addShader(VERTEX, _2d::bvh_vert);
        _programs(ProgramType::eBvhDraw).addShader(FRAGMENT, _2d::bvh_frag);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eFieldSlice).addShader(COMPUTE, _3d::field_slice_src);
        _programs(ProgramType::eBvhDraw).addShader(VERTEX, _3d::bvh_vert);
        _programs(ProgramType::eBvhDraw).addShader(FRAGMENT, _3d::bvh_frag);
      }
      _programs(ProgramType::eFlags).addShader(COMPUTE, flags_comp);
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
    glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());

    glm::uvec3 dispatch(1);

    // Specify buffers for cube data
    glNamedBufferStorage(_buffers(BufferType::eCubeVertices), 8 * sizeof(glm::vec4), cubeVertices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eCubeIndices), 24 * sizeof(unsigned), cubeIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eQuadVertices), 4 * sizeof(glm::vec2), quadVertices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eQuadLineIndices), 8 * sizeof(unsigned), quadLineIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eQuadTriangleIndices), 6 * sizeof(unsigned), quadTriangleIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eFlags), sizeof(uint) * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);

    // Specify vertex array object for instanced draw
    if constexpr (D == 2) {
      // Specify vertex array object for instanced draw
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eQuad), 
        _drawField ? _buffers(BufferType::eQuadTriangleIndices) : _buffers(BufferType::eQuadLineIndices));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 0, _buffers(BufferType::eQuadVertices), 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 1, buffers.node, 0, sizeof(uint));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 2, _buffers(BufferType::eFlags), 0, sizeof(uint));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 2, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 2);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eQuad), 1, 1, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eQuad), 2, 1, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 2, 2);
    } else if constexpr (D == 3) {
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 0, _buffers(BufferType::eCubeVertices), 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 1, buffers.node, 0, sizeof(uint));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 2, _buffers(BufferType::eFlags), 0, sizeof(uint));
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eCube), _buffers(BufferType::eCubeIndices));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 2, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 2);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 0, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eCube), 1, 1, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribIFormat(_vertexArrays(VertexArrayType::eCube), 2, 1, GL_UNSIGNED_INT, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 2, 2);
    }

    _nCube = layout.nNodes;
    _boundsBuffer = boundsBuffer;
    _pairsBuffer = pairsBuffer;
    _pairsHead = pairsHead;
    _isInit = true;
  }

  template <unsigned D>
  void FieldHierarchyRenderer<D>::destr()
  {
    if (!_isInit) {
      return;
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteVertexArrays(_vertexArrays.size(), _vertexArrays.data());
    glDeleteTextures(_textures.size(), _textures.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _hierarchy = nullptr;
    RenderComponent::destr();
  }

  template <unsigned D>
  void FieldHierarchyRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport)
  {
    if (!_isInit) {
      return;
    }
    if (!_hierarchy) {
      return;
    }
    
#ifdef INSERT_IMGUI
    ImGui::Begin("FieldHier Rendering");
    ImGui::Text("Tree bounding boxes");
    ImGui::Checkbox("Draw##1", &_drawCube);
    if (_drawCube) {
      ImGui::SliderFloat("Line width##1", &_cubeLineWidth, 0.5f, 16.f, "%1.1f");
      ImGui::SliderFloat("Line opacity##1", &_cubeOpacity, 0.f, 1.f, "%.2f");
      ImGui::Checkbox("Specific level", &_drawLvl);
      if (_drawLvl) {
        const auto layout = _hierarchy->layout();
        const uint lvlMin = 0;
        const uint lvlMax = layout.nLvls - 1;
        ImGui::SliderScalar("Level", ImGuiDataType_U32, &_bvhLvl, &lvlMin, &lvlMax);
      }
      if constexpr (D == 2) {
        ImGui::Checkbox("Draw", &_drawField);
        ImGui::Checkbox("Sum", &_doSum);
        if (_drawField != _drawFieldOld) {
          glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eQuad), 
            _drawField ? _buffers(BufferType::eQuadTriangleIndices) : _buffers(BufferType::eQuadLineIndices));
          _drawFieldOld = _drawField;
        }
      }
    }
    ImGui::End();
#endif // INSERT_IMGUI

    if (_drawCube && _drawFlags) {
      uint head = 0;
      glGetNamedBufferSubData(_pairsHead, 0, sizeof(uint), &head);
      glClearNamedBufferData(_buffers(BufferType::eFlags), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::eFlags);
      program.bind();

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _pairsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _pairsHead);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eFlags));

      glDispatchCompute(dr::ceilDiv(head, 256u), 1u, 1u);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glAssert("dbg::field_hierarchy::render::flags()");
    }
    
    if (_drawCube) {      
      const auto layout = _hierarchy->layout();

      // Bind program, uniforms
      auto &program = _programs(ProgramType::eBvhDraw);
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uCubeOpacity", _drawCube ? _cubeOpacity : 0.f);
      program.uniform1ui("lvl", _bvhLvl);
      program.uniform1ui("doLvl", _drawLvl);
      program.uniform1ui("doFlags", _drawFlags);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _hierarchy->buffers().field);

      // Set draw config
      glLineWidth(_cubeLineWidth);
      glDepthMask(GL_FALSE);

      // Perform instanced draw
      if constexpr (D == 2) {
        program.uniform1ui("doSum", _doSum);
        glBindVertexArray(_vertexArrays(VertexArrayType::eQuad));
        if (_drawField) {
          glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr, layout.nNodes);
        } else {
          glDrawElementsInstanced(GL_LINES, 8, GL_UNSIGNED_INT, nullptr, layout.nNodes);
        }
      } else if constexpr (D == 3) {
        glBindVertexArray(_vertexArrays(VertexArrayType::eCube));
        glDrawElementsInstanced(GL_LINES, 24, GL_UNSIGNED_INT, nullptr, layout.nNodes);
      }

      // Cleanup
      glDepthMask(GL_TRUE);
      program.release();
      glAssert("dbg::field_hierarchy::render::quads()");
    } // drawCube
    
#ifdef INSERT_IMGUI
    if constexpr (D == 3) {
      ImGui::Begin("FieldHier Slice");
      ImGui::Checkbox("Draw", &_drawField);
      if (_drawField) {
        const auto layout = _hierarchy->layout();
        const auto buffers = _hierarchy->buffers(); 

        if constexpr (D == 3) {
          ImGui::SliderFloat("Depth Value", &_fieldDepth, 0.f, 1.f);
        }
        const uint lvlMin = 1;
        const uint lvlMax = layout.nLvls - 1;
        ImGui::SliderScalar("Level", ImGuiDataType_U32, &_fieldLvl, &lvlMin, &lvlMax);
        
        ImVec2 outputDims = ImGui::GetWindowSize();
        if (_outputDims.x != outputDims.x) {
          _outputDims = ImVec2(outputDims.x, outputDims.x);
          glDeleteTextures(1, &_textures(TextureType::eOutput));
          glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eOutput));
          glTextureStorage2D(_textures(TextureType::eOutput), 1, GL_RGBA32F, _outputDims.x, _outputDims.y);
        }

        // Get a slice from the pixelBvh for a specified level
        {
          // Bind program and uniforms
          auto &program = _programs(ProgramType::eFieldSlice);
          program.bind();
          program.uniform2ui("dims", _outputDims.x, _outputDims.y);
          program.uniform1ui("lvl", _fieldLvl);
          if constexpr (D == 3) {
            program.uniform1ui("depth", static_cast<uint>(_fieldDepth * static_cast<float>(layout.dims.z)));
            program.uniform1ui("depthDims", layout.dims.z);
          }

          // Bind buffer objects
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.field);

          // Bind and clear output image
          glClearTexImage(_textures(TextureType::eOutput), 0, GL_RGBA, GL_FLOAT, nullptr);
          glBindImageTexture(0, _textures(TextureType::eOutput), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
          
          glDispatchCompute(dr::ceilDiv(static_cast<uint>(_outputDims.x), 16u), 
                            dr::ceilDiv(static_cast<uint>(_outputDims.y), 16u), 
                            1);
          glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        }
        
        // Draw slice using ImGui
        ImGui::Image((void *) (intptr_t) _textures(TextureType::eOutput), ImVec2(_outputDims.x, _outputDims.y));
      } // drawField
      ImGui::End();
    }
#endif // INSERT_IMGUI
  }

  // Explicit template instantiations for 2 and 3 dimensions
  template class FieldHierarchyRenderer<2>;
  template class FieldHierarchyRenderer<3>;
} // hdi::dbg