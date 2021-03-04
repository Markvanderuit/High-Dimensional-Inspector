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

#pragma once

#include "hdi/data/shader.h"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"

namespace hdi::dbg {
  template <unsigned D>
  class DualHierarchyRenderer : private RenderComponent {
  private:
    using uint = unsigned;

  public:
    DualHierarchyRenderer();
    ~DualHierarchyRenderer();

    void init(const dr::FieldHierarchy<D> &fieldHierarchy,
              const dr::EmbeddingBVH<D> &embeddingHierarchy,
              GLuint boundsBuffer,
              GLuint debugQueueBuffer,
              GLuint debugHeadBuffer);
    void destr();
    void render(glm::mat4 transform, glm::ivec4 viewport) override;

  private:
    enum class BufferType {
      eLineVertices,
      eLineIndices,
      eQuadVertices,

      Length
    };

    enum class VertexArrayType {
      eLine,
      eQuad,

      Length
    };

    enum class ProgramType {
      // eDot,
      eLines,
      eQuads,
      
      Length
    };

    GLuint _boundsBuffer;
    GLuint _debugQueueBuffer;
    GLuint _debugHeadBuffer;
    dr::EnumArray<BufferType, GLuint> _buffers;
    dr::EnumArray<VertexArrayType, GLuint> _vertexArrays;
    dr::EnumArray<ProgramType, ShaderProgram> _programs;

    // Hey look, a terrible idea
    const dr::FieldHierarchy<D> * _fieldHierarchyPtr;
    const dr::EmbeddingBVH<D> * _embeddingHierarchyPtr;

    // ImGui default settings
    bool _isActive = true; // global override
    uint _renderLvl = 7;
    uint _nodeIndex = 51;  // Index of interesting node
    float mousex = 0.55f;
    float mousey = 0.55f;

  public:
    bool nodeIndex() const {
      return _nodeIndex;
    }

    bool isInit() const {
      return _isInit;
    }

    bool isActive() const {
      return _isActive;
    }
  };
} // namespace hdi::dbg
