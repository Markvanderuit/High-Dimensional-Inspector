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

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/data/shader.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/embedding_bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/hierarchy/field_hierarchy.h"
#include "hdi/debug/renderer/field.hpp"
#include "hdi/debug/renderer/embedding_bvh.hpp" 
#include "hdi/debug/renderer/field_hierarchy.hpp" 
#include "hdi/debug/renderer/dual_hierarchy.hpp" 

namespace hdi::dr {
  /**
   * Field2dCompute
   * 
   * Class which computes the scalar and vector field components required for the tSNE 
   * minimization fully on the GPU. Manages subclasses and subcomponents such as tree
   * structures. Can perform full computation in O(p N) time, single tree approximation
   * in O(p log N) time, dual hierarchy approximation in O(log P log N) time.
   */
  class Field2dCompute {
    using Bounds = AlignedBounds<2>;
    using vec = dr::AlignedVec<2, float>;
    using uvec = dr::AlignedVec<2, uint>;

  public:
    Field2dCompute();
    ~Field2dCompute();

    void init(const TsneParameters& params,
                    GLuint positionBuffer,
                    GLuint boundsBuffer,
                    unsigned n);
    void destr();
    void compute(uvec dims, 
                 unsigned iteration, 
                 unsigned n,
                 GLuint positionBuffer, 
                 GLuint boundsBuffer, 
                 GLuint interpBuffer);

  private:
    enum class BufferType {
      eDispatch,

      // Queue + head for compact list of field pixels
      ePixels,
      ePixelsHead,
      ePixelsHeadReadback,
      
      // Queue + head with default list of node pairs for start of hierarchy traversal
      ePairsInit,
      ePairsInitHead,
      
      // Queue + head for list of node pairs which form large leaves during traversal
      ePairsLeaf,
      ePairsLeafHead,

      // Queue + head for list of node pairs where one has reached a leaf early
      ePairsDfs,
      ePairsDfsHead,

      // Queues + heads for list of node pairs for iterative traversal of hierarchy
      ePairsInput,
      ePairsOutput,
      ePairsInputHead,
      ePairsOutputHead,

#ifdef INSERT_DH_DEBUG
      // Queue + head for tracking all pairs interacting with one node
      eDebugQueue,
      eDebugHead,
#endif // INSERT_DH_DEBUG

      Length
    };

    enum class ProgramType {
      eDispatch,
      eStencil,
      eField,
      eInterp,

      // Single hierarchy program
      ePixels,
      eFieldBvh,

      // Dual hierarchy programs
      eFieldDual,
      eFieldDualLeaf,
      eFieldDualDfs,
      ePush,

      Length
    };

    enum class TimerType {
      eStencil, 
      eField, 
      eFieldLeaf,
      eFieldDfs,
      ePush,
      eInterp,
      
      Length
    };

    enum class TextureType {
      eStencil,
      eField,

      Length
    };

    // Hierarchies
    FieldHierarchy<2> _fieldHier;
    EmbeddingBVH<2> _embeddingBvh;

    // Pretty much all program, buffer and texture handles
    EnumArray<BufferType, GLuint> _buffers;
    EnumArray<ProgramType, ShaderProgram> _programs;
    EnumArray<TextureType, GLuint> _textures;
    EnumArray<TimerType, GLtimer> _timers;

    // Subcomponent for debug renderer
    dbg::FieldRenderer<2> _fieldRenderer;
    dbg::EmbeddingBVHRenderer<2> _embeddingBVHRenderer;
    dbg::FieldHierarchyRenderer<2> _fieldHierRenderer;
#ifdef INSERT_DH_DEBUG
    dbg::DualHierarchyRenderer<2> _dualHierRenderer;
#endif // INSERT_DH_DEBUG

    // Misc
    bool _isInit;
    uvec _dims;
    TsneParameters _params;
    utils::AbstractLog* _logger;
    bool _useEmbeddingBvh;
    bool _useFieldBvh;
    GLuint _stencilVao;
    GLuint _stencilFbo;
    uint _bvhRebuildIters;

  private:
    // Functions called by Field2dCompute::compute()
    void compactField(unsigned n,
                      GLuint positionsBuffer, 
                      GLuint boundsBuffer);
    void computeField(unsigned n,
                      unsigned iteration,
                      GLuint positionsBuffer,
                      GLuint boundsBuffer);
    void computeFieldBvh(unsigned n,
                         unsigned iteration,
                         GLuint positionsBuffer,
                         GLuint boundsBuffer);
    void computeFieldDualBvh(unsigned n,
                             unsigned iteration,
                             GLuint positionsBuffer,
                             GLuint boundsBuffer);
    void queryField(unsigned n,
                    unsigned iteration,
                    GLuint positionsBuffer,
                    GLuint boundsBuffer,
                    GLuint interpBuffer);

    void logIter(uint iteration, uint nPixels, bool fieldBvhActive) const;

  public:
    void setLogger(utils::AbstractLog* logger) {
      _logger = logger; 
    }

    void logTimerAverage() const;
  };
}