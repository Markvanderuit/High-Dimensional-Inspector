#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/embedding_bvh.hpp"
#include "hdi/debug/renderer/embedding_bvh_shaders_2d.h"
#include "hdi/debug/renderer/embedding_bvh_shaders_3d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"

GLSL(flags_comp, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  layout(binding = 0, std430) restrict readonly buffer NodeBuffer { vec4 nodeBuffer[]; };
  layout(binding = 1, std430) restrict readonly buffer DiamBuffer { vec3 diamBuffer[]; };
  layout(binding = 2, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
  layout(binding = 3, std430) restrict readonly buffer PosBuffer { vec3 pos; };
  layout(binding = 4, std430) restrict writeonly buffer FlagsBuffer { float flagsBuffer[]; };

  layout(location = 0) uniform uint nNodes;
  layout(location = 1) uniform float theta;   // Approximation param

  // Constants
  const uint kNode = BVH_KNODE_3D;
  const uint logk = BVH_LOGK_3D;
  const float theta2 = theta * theta;

  float approx(vec3 pos, uint idx) {
    vec4 node = nodeBuffer[idx];
    vec3 diam = diamBuffer[idx];

    // Squared eucl. distance between pos and node center
    vec3 t = pos - node.xyz;
    float t2 = dot(t, t);
    
    // Vector rejection of diam onto b
    vec3 b = normalize(abs(t) * vec3(-1, -1, 1));
    vec3 c = diam - b * dot(diam, b);

    const float pTheta = 1.f - dot(b, normalize(b * length(t) + 0.5f * c));

    // Approx if...
    const float appr = (pTheta * pTheta); 

    // const float appr = dot(c, c) / t2; 
    if (node.w < 2.f || appr < theta2) {
      return appr / theta2;
    } else {
      return 0.f;
    }
  }

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    // Check that invocation is inside range
    if (i >= nNodes) {
      return;
    }
    
    // Map pixel position to domain bounds
    vec3 domainPos = pos * bounds.range + bounds.min;

    // Test whether to flag this node for drawing
    float flag = approx(domainPos, i);
    bool parentFlag = false;
    if (flag > 0.f && i != 0) {
      uint _i = i;
      while (_i > 0 && parentFlag == false) {
        _i >>= logk;
        parentFlag = parentFlag || (approx(domainPos, _i) > 0.f);
      }
    }
    
    if (flag > 0.f && !parentFlag) {
      flagsBuffer[i] = flag;
    } else {
      flagsBuffer[i] = 0.f;
    }
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
  void EmbeddingBVHRenderer<D>::init(const dr::EmbeddingBVH<D> &bvh, GLuint boundsBuffer)
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
    glNamedBufferStorage(_buffers(BufferType::eFlags), sizeof(float) * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glm::vec4 v(0.5);
    glNamedBufferStorage(_buffers(BufferType::eFocusPosition), sizeof(glm::vec4), &v[0],
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT
    );

    if constexpr (D == 3) {
      // Specify vertex array object for instanced draw
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 0, _buffers(BufferType::eCubeVertices), 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 1, buffers.minb, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 2, buffers.node1, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 3, _buffers(BufferType::eFlags), 0, sizeof(float));
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eCube), _buffers(BufferType::eCubeIndices));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 2, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eCube), 3, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 2);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eCube), 3);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 0, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 1, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 2, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eCube), 3, 1, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 2, 2);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eCube), 3, 3);
      
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
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 0, _buffers(BufferType::eQuadVertices), 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 1, buffers.minb, 0, sizeof(glm::vec2));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 2, buffers.node1, 0, sizeof(glm::vec4));
      glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eQuad), 3, _buffers(BufferType::eFlags), 0, sizeof(float));
      glVertexArrayElementBuffer(_vertexArrays(VertexArrayType::eQuad), _buffers(BufferType::eQuadIndices));
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 2, 1);
      glVertexArrayBindingDivisor(_vertexArrays(VertexArrayType::eQuad), 3, 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 0);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 1);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 2);
      glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::eQuad), 3);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 1, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 2, 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::eQuad), 3, 1, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 0, 0);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 1, 1);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 2, 2);
      glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::eQuad), 3, 3);
      
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
      ImGui::SliderFloat("Theta##0", &_cubeTheta, 0.f, 1.f, "%.2f");
    }
    ImGui::Separator();
    ImGui::Text("Barnes-Hut approximation");
    ImGui::Checkbox("Draw##2", &_drawBarnesHut);
    if (_drawBarnesHut) {
      // Map focus position buffer for reading/writing so ImGUI components can touch it
      // Is it a bird? Is it a plane? No, it's a performance bottleneck!
      float *ptr = (float *) glMapNamedBuffer(_buffers(BufferType::eFocusPosition), GL_READ_WRITE);
      ImGui::SliderFloat("Line opacity##2", &_barnesHutOpacity, 0.f, 1.f, "%.2f");
      ImGui::SliderFloat3("Position", ptr, 0.f, 1.f);
      ImGui::SliderFloat("Theta##1", &_flagTheta, 0.f, 2.f);
      glUnmapNamedBuffer(_buffers(BufferType::eFocusPosition));
    }
    ImGui::End();

    // if (_bvh && _drawBarnesHut) {
    //   const auto layout = _bvh->layout();
    //   const auto buffers = _bvh->buffers(); 

    //   auto &program = _programs(ProgramType::eFlags);
    //   program.bind();

    //   // Set uniforms
    //   program.uniform1f("theta", _flagTheta);

    //   // Bind buffers
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _boundsBuffer);
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eFocusPosition));
    //   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eFlags));

    //   // Fire away
    //   glDispatchCompute(dr::ceilDiv(layout.nNodes, 256u), 1, 1);
      
    //   program.release();
    // }

    // if (_drawBarnesHut) {
    //   auto &program = _programs(ProgramType::eFocusPosDraw);
    //   program.bind();
    //   program.uniformMatrix4f("uTransform", &transform[0][0]);
    //   program.uniform1f("uOpacity", 1.f);
    //   glBindVertexArray(_vertexArrays(VertexArrayType::eFocusPos));
    //   glPointSize(32.f);
    //   glDrawArrays(GL_POINTS, 0, 1);
    //   program.release();
    // }
    
    // Set opengl state for line drawing operation and perform draw
    if (_drawCube || _drawBarnesHut) {      
      auto &program = _programs(ProgramType::eBvhDraw);
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uCubeOpacity", _drawCube ? _cubeOpacity : 0.f);
      program.uniform1f("uBarnesHutOpacity", _drawBarnesHut ? _barnesHutOpacity : 0.f);
      program.uniform1f("uTheta", _cubeTheta);
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