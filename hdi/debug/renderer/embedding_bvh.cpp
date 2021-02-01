#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/embedding_bvh.hpp"
#include "hdi/debug/renderer/embedding_bvh_shaders_2d.h"
#include "hdi/debug/renderer/embedding_bvh_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"

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
    flagsBuffer[pair.e] = 1;
  }
);

GLSL(focus_vert, 450,
  layout (location = 0) in vec3 vertex;
  layout (location = 0) uniform mat4 uTransform;

  void main() {
    gl_Position = uTransform * vec4(vertex, 1);
  }
);

GLSL(focus_frag, 450,
  layout (location = 0) out vec4 color;
  layout (location = 1) uniform float uOpacity;

  void main() {
    color = vec4(0, 0, 0, uOpacity);
  }
);

namespace {
  std::array<glm::vec2, 4> quadVertices = {
    glm::vec2(-0.5, -0.5),  // 0
    glm::vec2(0.5, -0.5),   // 1
    glm::vec2(0.5, 0.5),    // 2
    glm::vec2(-0.5, 0.5)    // 3
  };

  std::array<unsigned, 8> quadIndices = {
    0, 1,  1, 2,  2, 3,  3, 0
  };

  std::array<glm::vec4, 8> cubeVertices = {
    glm::vec4(-0.5, -0.5, -0.5, 0.f),  // 0
    glm::vec4(0.5, -0.5, -0.5, 0.f),   // 1
    glm::vec4(0.5, 0.5, -0.5, 0.f),    // 2
    glm::vec4(-0.5, 0.5, -0.5, 0.f),   // 3
    glm::vec4(-0.5, -0.5, 0.5, 0.f),   // 4
    glm::vec4(0.5, -0.5, 0.5, 0.f),    // 5
    glm::vec4(0.5, 0.5, 0.5, 0.f),     // 6
    glm::vec4(-0.5, 0.5, 0.5, 0.f)     // 7
  };

  std::array<unsigned, 24> cubeIndices = {
    0, 1,  1, 2,  2, 3,  3, 0,    // bottom
    0, 4,  1, 5,  2, 6,  3, 7,    // sides
    4, 5,  5, 6,  6, 7,  7, 4     // top
  };
}

namespace hdi::dbg {
  template <unsigned D>
  EmbeddingBVHRenderer<D>::EmbeddingBVHRenderer()
  : RenderComponent(1, false),
    _bvh(nullptr)
  { }

  template <unsigned D>
  EmbeddingBVHRenderer<D>::~EmbeddingBVHRenderer()
  {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void EmbeddingBVHRenderer<D>::init(const dr::EmbeddingBVH<D> &bvh, GLuint boundsBuffer, GLuint pairsBuffer, GLuint pairsHead)
  {
    RenderComponent::init();
    if (!_isInit) {
      return;
    }

    // This is gonna bite me
    _bvh = &bvh;
    
    // Query bvh for layout specs and buffer handles
    const auto layout = bvh.layout();
    const auto buffers = bvh.buffers(); 

    // Create and build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      if constexpr (D == 2) {
        _programs(ProgramType::eBvhDraw).addShader(VERTEX, _2d::bvh_vert);
        _programs(ProgramType::eBvhDraw).addShader(FRAGMENT, _2d::bvh_frag);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBvhDraw).addShader(VERTEX, _3d::bvh_vert);
        _programs(ProgramType::eBvhDraw).addShader(FRAGMENT, _3d::bvh_frag);
      }
      _programs(ProgramType::eFlags).addShader(COMPUTE, flags_comp);
      _programs(ProgramType::eFocusPosDraw).addShader(VERTEX, focus_vert);
      _programs(ProgramType::eFocusPosDraw).addShader(FRAGMENT, focus_frag);
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

    // Specify buffers for cube data
    glNamedBufferStorage(_buffers(BufferType::eCubeVertices), 8 * sizeof(glm::vec4), cubeVertices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eCubeIndices), 24 * sizeof(unsigned), cubeIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eQuadVertices), 4 * sizeof(glm::vec2), quadVertices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eQuadIndices), 8 * sizeof(unsigned), quadIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eFlags), sizeof(uint) * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glm::vec4 v(0.5);
    glNamedBufferStorage(_buffers(BufferType::eFocusPosition), sizeof(glm::vec4), &v[0],
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT
    );

    if constexpr (D == 3) {
      // Specify vertex array object for instanced draw
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eCube), _buffers(BufferType::eCubeIndices));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 0, _buffers(BufferType::eCubeVertices), 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 1, buffers.minb, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 2, buffers.node0, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 3, buffers.node1, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 4, _buffers(BufferType::eFlags), 0, sizeof(uint));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 2, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 3, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 4, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 2);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 3);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 4);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 0, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 1, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 2, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 3, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 4, 1, GL_UNSIGNED_INT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 2, 2);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 3, 3);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 4, 4);
      
      // Specify vertex array object for embedding positions
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::ePoints), 0, buffers.pos, 0, sizeof(glm::vec4));
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::ePoints), 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::ePoints), 0, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::ePoints), 0, 0);
      
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eFocusPos), 0, _buffers(BufferType::eFocusPosition), 0, sizeof(glm::vec4));
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eFocusPos), 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eFocusPos), 0, 3, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eFocusPos), 0, 0);
    } else if constexpr (D == 2) {
      // Specify vertex array object for instanced draw
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eQuad), _buffers(BufferType::eQuadIndices));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 0, _buffers(BufferType::eQuadVertices), 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 1, buffers.minb, 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 2, buffers.node0, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 3, buffers.node1, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 4, _buffers(BufferType::eFlags), 0, sizeof(uint));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 2, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 3, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 4, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 2);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 3);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 4);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 1, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 2, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 3, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 4, 1, GL_UNSIGNED_INT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 2, 2);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 3, 3);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 4, 4);
      
      // Specify vertex array object for embedding positions
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::ePoints), 0, buffers.pos, 0, sizeof(glm::vec2));
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::ePoints), 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::ePoints), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::ePoints), 0, 0);
      
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eFocusPos), 0, _buffers(BufferType::eFocusPosition), 0, sizeof(glm::vec2));
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eFocusPos), 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eFocusPos), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eFocusPos), 0, 0);
    }

    _nCube = layout.nNodes;
    _nPos = layout.nPos;
    _boundsBuffer = boundsBuffer;
    _pairsBuffer = pairsBuffer;
    _pairsHead = pairsHead;
    _isInit = true;
  }

  template <unsigned D>
  void EmbeddingBVHRenderer<D>::destr()
  {
    if (!_isInit) {
      return;
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteVertexArrays(_vertexArrays.size(), _vertexArrays.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _bvh = nullptr;
    RenderComponent::destr();
  }

  template <unsigned D>
  void EmbeddingBVHRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport)
  {
    if (!_isInit) {
      return;
    }
    if (!_bvh) {
      return;
    }
    
    ImGui::Begin("EmbeddingBVH Rendering");
    ImGui::Text("Tree bounding boxes");
    ImGui::Checkbox("Draw##1", &_drawCube);
    if (_drawCube) {
      ImGui::SliderFloat("Line width##1", &_cubeLineWidth, 0.5f, 16.f, "%1.1f");
      ImGui::SliderFloat("Line opacity##1", &_cubeOpacity, 0.f, 1.f, "%.2f");
      ImGui::Checkbox("Specific level", &_drawLvl);
      if (_drawLvl) {
        const auto layout = _bvh->layout();
        const uint lvlMin = 0;
        const uint lvlMax = layout.nLvls - 1;
        ImGui::SliderScalar("Level", ImGuiDataType_U32, &_bvhLvl, &lvlMin, &lvlMax);
      }
      if (_pairsBuffer != 0) {
        ImGui::Checkbox("Flagged only", &_drawFlags);
      }
    }
    ImGui::End();

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
    }
    
    // Set opengl state for line drawing operation and perform draw
    if (_drawCube) {      
      auto &program = _programs(ProgramType::eBvhDraw);
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uCubeOpacity", _drawCube ? _cubeOpacity : 0.f);
      program.uniform1ui("lvl", _bvhLvl);
      program.uniform1ui("doLvl", _drawLvl);
      program.uniform1ui("doFlags", _drawFlags);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glLineWidth(_cubeLineWidth);
      if constexpr (D == 2) {
        glBindVertexArray(_vertexArrays(VertexArrayType::eQuad));
        glDrawElementsInstanced(GL_LINES, 8, GL_UNSIGNED_INT, nullptr, _nCube);
      } else if constexpr (D == 3) {
        glBindVertexArray(_vertexArrays(VertexArrayType::eCube));
        glDrawElementsInstanced(GL_LINES, 24, GL_UNSIGNED_INT, nullptr, _nCube);
      }
      program.release();
    }
  }
  
  // Explicit template instantiations
  template class EmbeddingBVHRenderer<2>;
  template class EmbeddingBVHRenderer<3>;
}