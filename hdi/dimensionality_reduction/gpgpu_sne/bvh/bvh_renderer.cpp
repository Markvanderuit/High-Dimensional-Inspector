#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_renderer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_renderer_shaders.h"
#include <glm/gtx/transform.hpp>

namespace hdi::dr::bvh {
  BVHRenderer::BVHRenderer() { }

  BVHRenderer::~BVHRenderer() {
    if (_initialized) {
      destr();
    }
  }

  void BVHRenderer::init(const bvh::BVHLayout& bvhLayout, GLuint posBuffer) {
    // Create and build shader programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROG_TRIANGULATE].addShader(COMPUTE, triangulate_comp);
      _programs[PROG_BVH_DRAW].addShader(VERTEX, draw_bvh_vert);
      _programs[PROG_BVH_DRAW].addShader(FRAGMENT, draw_bvh_frag);
      _programs[PROG_EMB_DRAW].addShader(VERTEX, draw_emb_vert);
      _programs[PROG_EMB_DRAW].addShader(FRAGMENT, draw_emb_frag);
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
    glCreateFramebuffers(1, &fbo);

    // Specify buffer storage object
    unsigned nNode = bvhLayout.nNodes + bvhLayout.nLeaves;
    size_t nBytes = sizeof(float) * 4 * 24 * nNode;
    glNamedBufferStorage(_buffers[BUFF_LINES], nBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);

    // Specify vertex array object for bvh nodes
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_NODES], 0, _buffers[BUFF_LINES], 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_NODES], 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_NODES], 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_NODES], 0, 0);

    // Specify vertex array object for contained positions
    glVertexArrayVertexBuffer(_vertexArrays[VRAO_POINTS], 0, posBuffer, 0, 4 * sizeof(float));
    glEnableVertexArrayAttrib(_vertexArrays[VRAO_POINTS], 0);
    glVertexArrayAttribFormat(_vertexArrays[VRAO_POINTS], 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays[VRAO_POINTS], 0, 0);

    GL_ASSERT("BVHRenderer::init");
    _initialized = true;
  }

  void BVHRenderer::destr() {
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteVertexArrays(1, &vao);
    for (auto& program : _programs) {
      program.destroy();
    }

    GL_ASSERT("BVHRenderer::clean");
    _initialized = false;
  }

  void BVHRenderer::compute(const bvh::BVH& bvh, GLuint posBuffer, GLuint boundsBuffer) {
    // Query bvh for layout specs and buffer handle
    const auto& layout = bvh.layout();
    const auto& memr = bvh.memr(); 

    {
      auto &program = _programs[PROG_TRIANGULATE];
      program.bind();

      // Compute appropriate buffer range sizes
      GLsizeiptr float4Size = 4 * sizeof(float);
      GLsizeiptr nodeSize = float4Size * (layout.nNodes + layout.nLeaves);
      GLsizeiptr massSize = sizeof(unsigned) * (layout.nNodes + layout.nLeaves);

      // Compute appropriate buffer range offsets
      GLintptr nodeOffset = 0;
      GLintptr massOffset = nodeSize;

      // Bind buffer objects and ranges
      // memr.bindBuffer(BVHExtMemr::MemrType::eNode, 0);
      memr.bindBuffer(BVHExtMemr::MemrType::eMinB, 0);
      memr.bindBuffer(BVHExtMemr::MemrType::eMaxB, 1);
      memr.bindBuffer(BVHExtMemr::MemrType::eMass, 2);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, boundsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[BUFF_LINES]);

      // Set uniforms
      unsigned nNode = layout.nNodes + layout.nLeaves;
      program.uniform1ui("nNode", nNode);
      program.uniform1ui("nPos", layout.nPos);

      // Dispatch compute shader
      glDispatchCompute(ceilDiv(nNode, 256u), 1, 1);

      program.release();
      GL_ASSERT("BVHRenderer::compute::triangulate");
    }

    {
      constexpr std::array<float, 4> clearColor = { 0.0f, 0.0f, 0.0f, 0.f };
      constexpr float clearDepth = 1.f;

      // transform = glm::p
      glm::mat4 proj = glm::perspectiveFov(1.f, 980.f, 980.f, 0.001f, 1000.f);
      glm::mat4 view = glm::lookAt(glm::vec3(1.25), glm::vec3(0), glm::vec3(0, 1, 0));
      transform = proj * view;

      // Set opengl state for both drawing operations
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
      glViewport(0, 0, 1920, 1920);
      glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor.data());
      glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boundsBuffer);

      // Set opengl state for point drawing operation and perform draw
      {      
        auto &program = _programs[PROG_EMB_DRAW];
        program.bind();
        program.uniformMatrix4f("uTransform", &transform[0][0]);
        glBindVertexArray(_vertexArrays[VRAO_POINTS]);
        glPointSize(2.f);
        glDrawArrays(GL_POINTS, 0, layout.nPos);
        program.release();
      }

      // Set opengl state for node drawing operation and perform draw
      {
        auto &program = _programs[PROG_BVH_DRAW];
        program.bind();
        program.uniformMatrix4f("uTransform", &transform[0][0]);
        glBindVertexArray(_vertexArrays[VRAO_NODES]);
        glLineWidth(2.f);
        glDrawArrays(GL_LINES, 0, (layout.nNodes + layout.nLeaves) * 24);
        program.release();
      }
      
      // Reset opengl state to some baseline
      glBindVertexArray(0);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_BLEND);
      GL_ASSERT("BVHRenderer::compute::draw");
    }
  }
}