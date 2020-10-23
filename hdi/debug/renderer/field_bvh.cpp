#include <iostream>
#include <imgui.h>
#include "hdi/debug/renderer/field_bvh.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

GLSL(flags_comp, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  layout(binding = 0, std430) restrict readonly buffer FieldBuffer { vec4 fieldBuffer[]; };
  layout(binding = 1, std430) restrict writeonly buffer FlagsBuffer { float flagsBuffer[]; };
  
  layout(location = 0) uniform uint nNodes;
  layout(location = 1) uniform bool lvlOnly;
  layout(location = 2) uniform uint lvl;
  
  // Constants
  const uint kNode = BVH_KNODE_3D;
  const uint logk = BVH_LOGK_3D;

  uint findLvl(uint i) {
    uint lvl = 0u;
    do { lvl++; } while ((i = ( i - 1) >> logk) > 0);
    return lvl;
  }

  void main() {
    const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    // Check that invocation is inside range
    if (i >= nNodes) {
      return;
    }

    if (fieldBuffer[i] != vec4(0)) {
      if (lvlOnly) {
        // Only render a specific level;
        if (findLvl(i) == lvl) {
          flagsBuffer[i] = 1.f;
        } else {
          flagsBuffer[i] = 0.f;
        }
      } else {
        flagsBuffer[i] = 1.f;
      }
    } else {
      flagsBuffer[i] = 0.f;
    }
  }
);

GLSL(divide_dispatch_src, 450,
  layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 0, std430) restrict readonly buffer Value { uint value; };
  layout(binding = 1, std430) restrict writeonly buffer Disp { uvec3 dispatch; };
  layout(location = 0) uniform uint div;

  void main() {
    dispatch = uvec3((value + div - 1) / div, 1, 1);
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

  layout(location = 0) in vec3 vert; // instanced
  layout(location = 1) in vec3 node0;
  layout(location = 2) in vec3 node1;
  layout(location = 3) in float flag;
  layout(location = 0) out vec3 pos;
  layout(location = 1) out float theta;
  layout(location = 2) out float fflag;
  layout(location = 0) uniform mat4 uTransform;
  layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

  void main() {
    fflag = flag;

    // Obtain normalized [0, 1] boundary values
    vec3 _minb = (node0 - bounds.min) * bounds.invRange;
    vec3 _diam = node1 * bounds.invRange;

    // Generate position from instanced data
    pos = _minb + (0.5 + vert) * _diam;

    // Compute uhh, anisotropy of box
    theta = dot(normalize(_diam), normalize(vec3(1)));

    // Apply camera transformation to output data
    gl_Position = uTransform * vec4(pos, 1);
  }
);

GLSL(bvh_frag, 450,
  layout(location = 0) in vec3 pos;
  layout(location = 1) in float theta;
  layout(location = 2) in float flag;
  layout(location = 0) out vec4 color;
  layout(location = 1) uniform float uCubeOpacity;
  layout(location = 2) uniform float uBarnesHutOpacity;
  layout(location = 3) uniform float uTheta;
  
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

GLSL(field_slice_src, 450, 
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer VPixe { uvec3 posBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

  // Uniform values
  layout(location = 0) uniform uint outputLvl;
  layout(location = 1) uniform uint outputDepth;
  
  // Constants
  const uint kNode = BVH_KNODE_3D;
  const uint logk = BVH_LOGK_3D;

  uint findLvl(uint i) {
    uint lvl = 0u;
    do { lvl++; } while ((i = ( i - 1) >> logk) > 0);
    return lvl;
  }

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (j >= leafHead) {
      return;
    }

    const uint i = leafBuffer[j];
    // const uvec3 px = posBuffer[i];
    // if (px.z != outputDepth) {
    //   return;
    // }

    uint _i = i;
    uint lvl = findLvl(_i);

    while (outputLvl < lvl) {
      lvl--;
      _i = (_i - 1) >> logk;
    }

    vec4 field = fieldBuffer[_i];
    // vec4 field = vec4(0);
    // for (int i = int(lvl); i > 0; i--) {
    //   field += fieldBuffer[_i];
    //   _i = (_i - 1) >> logk;
    // }

    const uint begin = uint(node1Buffer[i].w);
    const uint extent = uint(node0Buffer[i].w);
    for (uint k = begin; k < begin + extent; k++) {
      uvec3 px = posBuffer[k].xyz;
      if (px.z == outputDepth) {
        imageStore(fieldImage, ivec2(px.xy), field);
      }
    }
  }
);

GLSL(field_resample_src, 450,
  layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
  layout(binding = 0) uniform sampler2D inputSampler;
  layout(binding = 0, rgba32f) restrict writeonly uniform image2D outputImage;
  layout(location = 0) uniform uvec2 outputSize;

  void main() {
    // Check that invocation is inside input texture bounds
    const uvec2 i = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
    if (min(i, outputSize - 1) != i) {
      return;
    }

    // Sample input texture
    vec4 field = texture(inputSampler, (vec2(i) + 0.5) / vec2(outputSize));

    // Write to output texture
    vec3 v = vec3(0.5f);
    if (field != vec4(0)) {
      v += 0.5 * normalize(field.yzw);
    }
    imageStore(outputImage, ivec2(i), vec4(v, 1));
  }
);

namespace {
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
  FieldBVHRenderer<D>::FieldBVHRenderer()
  : RenderComponent(5, false),
    _bvh(nullptr)
  { }

  template <unsigned D>
  FieldBVHRenderer<D>::~FieldBVHRenderer()
  {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void FieldBVHRenderer<D>::init(const dr::FieldBVH<3> &bvh, GLuint boundsBuffer)
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
      _programs(ProgramType::eFlags).addShader(COMPUTE, flags_comp);
      _programs(ProgramType::eBvhDraw).addShader(VERTEX, bvh_vert);
      _programs(ProgramType::eBvhDraw).addShader(FRAGMENT, bvh_frag);
      _programs(ProgramType::eFieldSlice).addShader(COMPUTE, field_slice_src);
      _programs(ProgramType::eFieldResample).addShader(COMPUTE, field_resample_src);
      _programs(ProgramType::eDispatch).addShader(COMPUTE, divide_dispatch_src);
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
    glNamedBufferStorage(_buffers(BufferType::eFlags), sizeof(float) * (layout.nNodes), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec3), &dispatch[0], 0);

    // Specify vertex array object for instanced draw
    glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 0, _buffers(BufferType::eCubeVertices), 0, sizeof(glm::vec4));
    glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::eCube), 1, buffers.node0, 0, sizeof(glm::vec4));
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
    glVertexArrayVertexBuffer(_vertexArrays(VertexArrayType::ePoints), 0, buffers.pixel, 0, sizeof(glm::vec4));
    glEnableVertexArrayAttrib(_vertexArrays(VertexArrayType::ePoints), 0);
    glVertexArrayAttribFormat(_vertexArrays(VertexArrayType::ePoints), 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(_vertexArrays(VertexArrayType::ePoints), 0, 0);

    _nCube = layout.nNodes;
    _boundsBuffer = boundsBuffer;
    _isInit = true;
  }

  template <unsigned D>
  void FieldBVHRenderer<D>::destr()
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
    _bvh = nullptr;
    RenderComponent::destr();
  }

  template <unsigned D>
  void FieldBVHRenderer<D>::render(glm::mat4 transform, glm::ivec4 viewport)
  {
    if (!_isInit) {
      return;
    }
    if (!_bvh) {
      return;
    }
    
    ImGui::Begin("PixelBVH Rendering");

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
    ImGui::Checkbox("Level only", &_drawSpecificLevel);

    ImGui::Separator();

    ImGui::Text("Field");
    ImGui::Checkbox("Draw##3", &_drawField);


    if (_drawBarnesHut) {
      const auto layout = _bvh->layout();
      const auto buffers = _bvh->buffers(); 

      auto &program = _programs(ProgramType::eFlags);
      program.bind();

      // Set uniforms
      program.uniform1ui("nNodes", layout.nNodes);
      program.uniform1ui("lvlOnly", _drawSpecificLevel ? 1u : 0u);
      program.uniform1ui("lvl", _fieldLvl);

      // Bind buffers
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eFlags));

      // Fire away
      glDispatchCompute(dr::ceilDiv(layout.nNodes, 256u), 1, 1);
      
      program.release();
    }
    
    // Set opengl state for line drawing operation and perform draw
    if (_drawCube || _drawBarnesHut) {      
      auto &program = _programs(ProgramType::eBvhDraw);
      program.bind();
      program.uniformMatrix4f("uTransform", &transform[0][0]);
      program.uniform1f("uCubeOpacity", _drawCube ? _cubeOpacity : 0.f);
      program.uniform1f("uBarnesHutOpacity", _drawBarnesHut ? _barnesHutOpacity : 0.f);
      program.uniform1f("uTheta", _cubeTheta);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _boundsBuffer);
      glBindVertexArray(_vertexArrays(VertexArrayType::eCube));
      glLineWidth(_cubeLineWidth);
      glDrawElementsInstanced(GL_LINES, 24, GL_UNSIGNED_INT, nullptr, _nCube);
      program.release();
    }

    if (_drawField) {
      const auto layout = _bvh->layout();
      const auto buffers = _bvh->buffers(); 

      if constexpr (D == 3) {
        ImGui::SliderFloat("Depth Value", &_fieldDepth, 0.f, 1.f);
      }
      const uint lvlMin = 0;
      const uint lvlMax = layout.nLvls - 1;
      ImGui::SliderScalar("Level", ImGuiDataType_U32, &_fieldLvl, &lvlMin, &lvlMax);
      
      auto fieldDims = layout.dims;
      ImVec2 outputDims = ImGui::GetWindowSize();

      if (_fieldDims.x != fieldDims.x || _fieldDims.y != fieldDims.y) {
        _fieldDims = ImVec2(fieldDims.x, fieldDims.y);
        glDeleteTextures(1, &_textures(TextureType::eField));
        glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eField));
        glTextureStorage2D(_textures(TextureType::eField), 1, GL_RGBA32F, _fieldDims.x, _fieldDims.y);
      }

      if (_outputDims.x != outputDims.x) {
        _outputDims = ImVec2(outputDims.x, outputDims.x);
        glDeleteTextures(1, &_textures(TextureType::eOutput));
        glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eOutput));
        glTextureStorage2D(_textures(TextureType::eOutput), 1, GL_RGBA32F, _outputDims.x, _outputDims.y);
      }


      // Get a slice from the pixelBvh for a specified level
      {
        // Divide contents of BufferType::eLeafHead by workgroup size
        {
          auto &program = _programs(ProgramType::eDispatch);
          program.bind();
          program.uniform1ui("div", 256);

          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.leafHead);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
          
          glDispatchCompute(1, 1, 1);
        }

        glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

        // Bind program and uniforms
        auto &program = _programs(ProgramType::eFieldSlice);
        program.bind();
        program.uniform1ui("outputLvl", _fieldLvl);
        program.uniform1ui("outputDepth", static_cast<uint>(_fieldDepth * static_cast<float>(layout.dims.z)));

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.pixel);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers.field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers.leafFlags);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buffers.leafHead);

        // Bind output image
        glBindImageTexture(0, _textures(TextureType::eField), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        // Dispatch shader
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      // Resample the slice to the output texture
      {
        // Bind program and uniforms
        auto &program = _programs(ProgramType::eFieldResample);
        program.bind();
        program.uniform2ui("outputSize", _outputDims.x, _outputDims.y);

        glBindTextureUnit(0, _textures(TextureType::eField));
        glBindImageTexture(0, _textures(TextureType::eOutput), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(dr::ceilDiv(static_cast<unsigned>(_outputDims.x), 16u), 
                          dr::ceilDiv(static_cast<unsigned>(_outputDims.y), 16u), 
                          1);
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
      }
      ImGui::Image((void *) (intptr_t) _textures(TextureType::eOutput), ImVec2(_outputDims.x, _outputDims.y));
    }

    ImGui::End();
  }

  // Explicit template instantiations for 2 and 3 dimensions
  template class FieldBVHRenderer<2>;
  template class FieldBVHRenderer<3>;
} // hdi::dbg