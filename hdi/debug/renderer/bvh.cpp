#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/bvh.hpp"

#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

template <typename Int> 
inline
Int ceilDiv(Int n, Int div) {
  return (n + div - 1) / div;
}

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
  layout(binding = 5, std430) restrict writeonly buffer LineBuffer { vec3 lineBuffer[]; };

  layout(location = 0) uniform uint nNodes;
  layout(location = 1) uniform uint kNode;
  layout(location = 2) uniform float theta;   // Approximation param

  // Constants
  const uint logk = uint(log2(kNode));
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

    // Approx if...
    float appr = dot(c, c) / t2; 
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
      lineBuffer[2 * i] = pos;
      lineBuffer[2 * i + 1] = (nodeBuffer[i].xyz - bounds.min) * bounds.invRange;
    } else {
      flagsBuffer[i] = 0.f;
      lineBuffer[2 * i] = vec3(100000);
      lineBuffer[2 * i + 1] = vec3(100000);
    }
  }
);

GLSL(bvh_vert, 450,
  // Wrapper structure for BoundsBuffer data
  struct Bounds {
    vec3 min;
    vec3 max;
    vec3 range;
    vec3 invRange;
  };

  layout (location = 0) in vec3 vert; // instanced
  layout (location = 1) in vec3 minb;
  layout (location = 2) in vec3 diam;
  layout (location = 3) in float flag;
  layout (location = 0) out vec3 pos;
  layout (location = 1) out float theta;
  layout (location = 2) out float fflag;

  layout (location = 0) uniform mat4 uTransform;
  layout (binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

  void main() {
    fflag = flag;

    // Obtain normalized [0, 1] boundary values
    vec3 _minb = (minb - bounds.min) * bounds.invRange;
    vec3 _diam = diam * bounds.invRange;

    // Generate position from instanced data
    pos = _minb + (0.5 + vert) * _diam;

    // Compute uhh, anisotropy of box
    theta = dot(normalize(_diam), normalize(vec3(1)));

    // Apply camera transformation to output data
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(bvh_frag, 450,
  layout (location = 0) in vec3 pos;
  layout (location = 1) in float theta;
  layout (location = 2) in float flag;
  layout (location = 0) out vec4 color;
  layout (location = 1) uniform float uCubeOpacity;
  layout (location = 2) uniform float uBarnesHutOpacity;
  layout (location = 3) uniform float uTheta;
  
  void main() {
    if (uBarnesHutOpacity > 0.f && flag > 0.f) {
      if (theta < uTheta) {
        color =  vec4(1, 0, 0, 1);
      } else {
        color =  vec4(0, 0, 0, uBarnesHutOpacity);
      }
    } else if (uCubeOpacity > 0.f) {
      if (theta < uTheta) {
        color =  vec4(1, 0, 0, 1);
      } else {
        color =  vec4(normalize(pos), uCubeOpacity);
      }
    } else {
      discard;
    }
  }
);

GLSL(order_vert, 450,
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
    vec3 _fpos = (vertex.xyz - bounds.min) * bounds.invRange;
    ivec3 _ipos = ivec3(clamp(_fpos * 1024.f, 0.f, 1023.f));

    pos = vec3(_ipos) / 1024.f;
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(order_frag, 450,
  layout (location = 0) in vec3 pos;
  layout (location = 0) out vec4 color;
  layout (location = 1) uniform float uOpacity;

  void main() {
    color = vec4(0.2 + 0.8 * normalize(pos), uOpacity);
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

namespace hdi::dbg {
  BvhRenderer::BvhRenderer()
  : RenderComponent(1, false),
    _bvh(nullptr)
  { }

  BvhRenderer::~BvhRenderer()
  {
    if (_isInit) {
      destr();
    }
  }

  void BvhRenderer::init(const dr::BVH<3> &bvh, GLuint boundsBuffer)
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
      _programs[PROG_FLAGS].addShader(COMPUTE, flags_comp);
      _programs[PROG_BVH_INSTANCED_DRAW].addShader(VERTEX, bvh_vert);
      _programs[PROG_BVH_INSTANCED_DRAW].addShader(FRAGMENT, bvh_frag);
      _programs[PROG_ORDER_DRAW].addShader(VERTEX, order_vert);
      _programs[PROG_ORDER_DRAW].addShader(FRAGMENT, order_frag);
      _programs[PROG_FOCUS_POS_DRAW].addShader(VERTEX, focus_vert);
      _programs[PROG_FOCUS_POS_DRAW].addShader(FRAGMENT, focus_frag);
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
    glNamedBufferStorage(_buffers[BUFF_CUBE_VERT], 8 * sizeof(glm::vec4), cubeVertices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers[BUFF_CUBE_IDX], 24 * sizeof(unsigned), cubeIndices.data(), GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers[BUFF_FLAGS], sizeof(float) * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glm::vec4 v(0.5);
    glNamedBufferStorage(_buffers[BUFF_FOCUS_POS], sizeof(glm::vec4), &v[0],
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT
    );
    glNamedBufferStorage(_buffers[BUFF_FOCUS_LINES], sizeof(glm::vec4) * 2 * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);

    // Specify vertex array object for instanced draw
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_CUBE], 0, _buffers[BUFF_CUBE_VERT], 0, sizeof(glm::vec4));
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_CUBE], 1, buffers.minb, 0, sizeof(glm::vec4));
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_CUBE], 2, buffers.node1, 0, sizeof(glm::vec4));
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_CUBE], 3, _buffers[BUFF_FLAGS], 0, sizeof(float));
    glVertexArrayElementBuffer(_vertexArrays[VRAO_CUBE], _buffers[BUFF_CUBE_IDX]);
    glVertexArrayBindingDivisor(_vertexArrays[VRAO_CUBE], 0, 0);
    glVertexArrayBindingDivisor(_vertexArrays[VRAO_CUBE], 1, 1);
    glVertexArrayBindingDivisor(_vertexArrays[VRAO_CUBE], 2, 1);
    glVertexArrayBindingDivisor(_vertexArrays[VRAO_CUBE], 3, 1);
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_CUBE], 0);
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_CUBE], 1);
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_CUBE], 2);
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_CUBE], 3);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_CUBE], 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_CUBE], 1, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_CUBE], 2, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_CUBE], 3, 1, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_CUBE], 0, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_CUBE], 1, 1);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_CUBE], 2, 2);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_CUBE], 3, 3);

    // Specify vertex array object for embedding positions
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_POINTS], 0, buffers.pos, 0, sizeof(glm::vec4));
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_POINTS], 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_POINTS], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_POINTS], 0, 0);
    
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_FOCUS_POS], 0, _buffers[BUFF_FOCUS_POS], 0, sizeof(glm::vec4));
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_FOCUS_POS], 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_FOCUS_POS], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_FOCUS_POS], 0, 0);
    
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_FOCUS_LINES], 0, _buffers[BUFF_FOCUS_LINES], 0, sizeof(glm::vec4));
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_FOCUS_LINES], 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_FOCUS_LINES], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_FOCUS_LINES], 0, 0);

    _nCube = layout.nNodes;
    _nPos = layout.nPos;
    _boundsBuffer = boundsBuffer;
    _isInit = true;
  }

  void BvhRenderer::destr()
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

  void BvhRenderer::render(glm::mat4 transform, glm::ivec4 viewport)
  {
    if (!_isInit) {
      return;
    }
    if (!_bvh) {
      return;
    }

    // Map focus position buffer for reading/writing so ImGUI components can touch it
    float *ptr = (float *) glMapNamedBuffer(_buffers[BUFF_FOCUS_POS], GL_READ_WRITE);
    
    ImGui::Begin("BVH Rendering");
    ImGui::Text("Z-Order Curve");
    ImGui::Checkbox("Draw##0", &_drawOrder);
    ImGui::SliderFloat("Line width##0", &_orderLineWidth, 0.5f, 16.f, "%1.1f");
    ImGui::SliderFloat("Line opacity##0", &_orderOpacity, 0.05f, 1.f, "%.2f");
    ImGui::Separator();
    ImGui::Text("Tree bounding boxes");
    ImGui::Checkbox("Draw##1", &_drawCube);
    ImGui::SliderFloat("Line width##1", &_cubeLineWidth, 0.5f, 16.f, "%1.1f");
    ImGui::SliderFloat("Line opacity##1", &_cubeOpacity, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Theta##0", &_cubeTheta, 0.f, 1.f, "%.2f");
    ImGui::Separator();
    ImGui::Text("Barnes-Hut approximation");
    ImGui::Checkbox("Draw##2", &_drawBarnesHut);
    ImGui::SliderFloat("Line opacity##2", &_barnesHutOpacity, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat3("Position", ptr, 0.f, 1.f);
    ImGui::SliderFloat("Theta##1", &_flagTheta, 0.f, 2.f);
    ImGui::End();

    // Unmap focus position buffer
    glUnmapNamedBuffer(_buffers[BUFF_FOCUS_POS]);

    if (_bvh && _drawBarnesHut) {
      const auto layout = _bvh->layout();
      const auto buffers = _bvh->buffers(); 

      auto &program = _programs[PROG_FLAGS];
      program.bind();

      // Set uniforms
      program.uniform1ui("nNodes", layout.nNodes);
      program.uniform1ui("kNode", layout.nodeFanout);
      program.uniform1f("theta", _flagTheta);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFF_FOCUS_POS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[BUFF_FLAGS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers[BUFF_FOCUS_LINES]);

      // Fire away
      glDispatchCompute(ceilDiv(layout.nNodes, 256u), 1, 1);
      
      program.release();
    }

    if (_drawBarnesHut) {
      auto &program = _programs[PROG_FOCUS_POS_DRAW];
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uOpacity", 1.f);
      glBindVertexArray(_vertexArrays[VRAO_FOCUS_POS]);
      glPointSize(32.f);
      glDrawArrays(GL_POINTS, 0, 1);
      program.release();
    }

    /* {
      const auto& layout = _bvh->layout();
      auto &program = _programs[PROG_FOCUS_POS_DRAW];
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uOpacity", 0.25f);
      glBindVertexArray(_vertexArrays[VRAO_FOCUS_LINES]);
      glLineWidth(1.f);
      glDrawArrays(GL_LINES, 0, layout.nNodes - layout.nLeaves);
      program.release();
    } */

    // Set opengl state for line drawing operation and perform draw
    if (_drawOrder) {      
      auto &program = _programs[PROG_ORDER_DRAW];
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uOpacity", _orderOpacity);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindVertexArray(_vertexArrays[VRAO_POINTS]);
      glLineWidth(_orderLineWidth);
      glDrawArrays(GL_LINE_STRIP, 0, _nPos);
      program.release();
    }
    
    // Set opengl state for line drawing operation and perform draw
    if (_drawCube || _drawBarnesHut) {      
      auto &program = _programs[PROG_BVH_INSTANCED_DRAW];
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uCubeOpacity", _drawCube ? _cubeOpacity : 0.f);
      program.uniform1f("uBarnesHutOpacity", _drawBarnesHut ? _barnesHutOpacity : 0.f);
      program.uniform1f("uTheta", _cubeTheta);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindVertexArray(_vertexArrays[VRAO_CUBE]);
      glLineWidth(_cubeLineWidth);
      glDrawElementsInstanced(GL_LINES, 24, GL_UNSIGNED_INT, nullptr, _nCube);
      program.release();
    }
  }
}