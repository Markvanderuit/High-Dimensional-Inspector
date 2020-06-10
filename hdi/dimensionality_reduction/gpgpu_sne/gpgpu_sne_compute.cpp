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

#include <algorithm>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute_shaders_3d.h"

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

constexpr bool doAdaptiveResolution = true;
constexpr unsigned minFieldSize = 64;       // 5
constexpr unsigned fixedFieldSize = 256;    // 40
constexpr unsigned fixedDepthSize = 16;     // ...?
constexpr float pixelRatio = 2.0f;
constexpr float functionSupport = 6.5f;
constexpr float boundsPadding = 0.1f;

namespace hdi::dr {
  template <unsigned D>
  GpgpuSneCompute<D>::GpgpuSneCompute()
  : _isInit(false), _logger(nullptr)
  { }

  template <unsigned D>
  GpgpuSneCompute<D>::~GpgpuSneCompute()
  {
    if (_isInit) {
      destr();
    }
  }

  template <unsigned D>
  void GpgpuSneCompute<D>::init(const Embedding *embedding,
                                const TsneParameters &params,
                                const SparseMatrix &P)
  {
    _params = params;

    // Initialize shader program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      // Set shaders depending on required dimension
      if constexpr (D == 2) {
        _programs[PROG_BOUNDS].addShader(COMPUTE, _2d::bounds_src);
        _programs[PROG_SUM_Q].addShader(COMPUTE, _2d::sumq_src);
        _programs[PROG_POSITIVE_FORCES].addShader(COMPUTE, _2d::positive_forces_src);
        _programs[PROG_GRADIENTS].addShader(COMPUTE, _2d::gradients_src);
        _programs[PROG_UPDATE].addShader(COMPUTE, _2d::update_src);
        _programs[PROG_CENTER].addShader(COMPUTE, _2d::center_src);
      } else if constexpr (D == 3) {
        _programs[PROG_BOUNDS].addShader(COMPUTE, _3d::bounds_src);
        _programs[PROG_SUM_Q].addShader(COMPUTE, _3d::sumq_src);
        _programs[PROG_POSITIVE_FORCES].addShader(COMPUTE, _3d::positive_forces_src);
        _programs[PROG_GRADIENTS].addShader(COMPUTE, _3d::gradients_src);
        _programs[PROG_UPDATE].addShader(COMPUTE, _3d::update_src);
        _programs[PROG_CENTER].addShader(COMPUTE, _3d::center_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
    }
    
    // Initialize buffer objects
    {
      const unsigned n = embedding->numDataPoints();
      const auto *data = embedding->getContainer().data();
      const LinearProbabilityMatrix LP = linearizeProbabilityMatrix(embedding, P);
      const unsigned _D = (D > 2) ? 4 : D; // aligned D
      const std::vector<float> zeroes(_D * n, 0);
      const std::vector<float> ones(_D * n, 1);

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers[BUFF_POSITION], n * sizeof(vec), data, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers[BUFF_INTERP_FIELDS], n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers[BUFF_SUM_Q], 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers[BUFF_SUM_Q_REDUCE_ADD], 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers[BUFF_NEIGHBOUR], LP.neighbours.size() * sizeof(uint32_t), LP.neighbours.data(), 0);
      glNamedBufferStorage(_buffers[BUFF_PROBABILITIES], LP.probabilities.size() * sizeof(float), LP.probabilities.data(), 0);
      glNamedBufferStorage(_buffers[BUFF_INDEX], LP.indices.size() * sizeof(int), LP.indices.data(), 0);
      glNamedBufferStorage(_buffers[BUFF_POSITIVE_FORCES], n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers[BUFF_GRADIENTS], n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers[BUFF_PREV_GRADIENTS], n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers[BUFF_GAIN], n * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers[BUFF_BOUNDS], 4 * sizeof(vec), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers[BUFF_BOUNDS_REDUCE_ADD], 256 * sizeof(vec), ones.data(), 0);
    }

    // Initialize field computation subcomponent
    if constexpr (D == 2) {
      _2dFieldComputation.init(params, _buffers[BUFF_POSITION], _buffers[BUFF_BOUNDS], embedding->numDataPoints());
    } else if constexpr (D == 3) {
      _3dFieldComputation.init(params, _buffers[BUFF_POSITION], _buffers[BUFF_BOUNDS], embedding->numDataPoints());
    }

    // Initialize embedding renderer subcomponent
    if constexpr (D == 2) {
      // TODO init scatterplot renderer...
    } else if constexpr (D == 3) {
      _embeddingRenderer.init(
        embedding->numDataPoints(), 
        _buffers[BUFF_POSITION], 
        _buffers[BUFF_BOUNDS]
      );
    }

    INIT_TIMERS();
    
    ASSERT_GL("GpgpuSneCompute::init()");
    _isInit = true;
  }

  template <unsigned D>
  void GpgpuSneCompute<D>::destr()
  {
    // Destroy field computation subcomponent
    if constexpr (D == 2) {
      _2dFieldComputation.destr();
    } else if constexpr (D == 3) {
      _3dFieldComputation.destr();
    }

    // Destroy embedding renderer subcomponent
    if constexpr (D == 2) {
      // TODO destr scatterplot renderer...
    } else if constexpr (D == 3) {
      _embeddingRenderer.destr();
    }

    // Destroy own components
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    DSTR_TIMERS();

    ASSERT_GL("GpgpuSneCompute::destr()");
    _isInit = false;
  }

  // Specialization of compute function for 2 dimensions
  template <unsigned D>
  void GpgpuSneCompute<D>::compute(Embedding* embeddingPtr,
                                   float exaggeration,
                                   unsigned iteration,
                                   float mult)
  {
    const unsigned int n = embeddingPtr->numDataPoints();

    // Compute embedding bounds
    {
      TICK_TIMER(TIMR_BOUNDS);

      auto &program = _programs[PROG_BOUNDS];
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("padding", boundsPadding);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_BOUNDS_REDUCE_ADD]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFF_BOUNDS]);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform1ui("iter", 0u);
      glDispatchCompute(128, 1, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);

      TOCK_TIMER(TIMR_BOUNDS);
      ASSERT_GL("GpgpuSneCompute::compute::bounds()");
    }
    
    // Compute field approximation
    {
      // Copy bounds back to host
      glGetNamedBufferSubData(
        _buffers[BUFF_BOUNDS],
        0,
        sizeof(Bounds),
        &_bounds
      );

      // Determine field texture dimensions
      vec range = _bounds.range();
      uvec dims = doAdaptiveResolution
                ? glm::max(uvec(range * pixelRatio), minFieldSize)
                : uvec(fixedFieldSize);

      // Perform field approximation
      if constexpr (D == 2) {
        _2dFieldComputation.compute(
          dims, functionSupport, iteration, n,
          _buffers[BUFF_POSITION],
          _buffers[BUFF_BOUNDS],
          _buffers[BUFF_INTERP_FIELDS],
          _bounds
        );
      } else if constexpr (D == 3) {
        _3dFieldComputation.compute(
          dims, functionSupport, n,
          _buffers[BUFF_POSITION],
          _buffers[BUFF_BOUNDS],
          _buffers[BUFF_INTERP_FIELDS],
          _bounds
        );
      }
    }

    // Calculate sum over Q
    {
      TICK_TIMER(TIMR_SUM_Q);

      auto &program = _programs[PROG_SUM_Q];
      program.bind();
      program.uniform1ui("nPoints", n);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_SUM_Q_REDUCE_ADD]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFF_SUM_Q]);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform1ui("iter", 0u);
      glDispatchCompute(128, 1, 1);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);

      TOCK_TIMER(TIMR_SUM_Q);
      ASSERT_GL("GpgpuSneCompute::compute::sum_q()");
    }

    // Compute resulting gradient
    {
      TICK_TIMER(TIMR_GRADIENTS);

      {
        auto &program = _programs[PROG_POSITIVE_FORCES];
        program.bind();
        program.uniform1ui("nPoints", n);
        program.uniform1f("invNPoints", 1.f / static_cast<float>(n));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_POSITION]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_NEIGHBOUR]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFF_PROBABILITIES]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFF_INDEX]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[BUFF_POSITIVE_FORCES]);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glDispatchCompute(n, 1, 1);
      }

      ASSERT_GL("GpgpuSneCompute::compute::pos_forces()");

      {
        auto &program = _programs[PROG_GRADIENTS];
        program.bind();
        program.uniform1ui("nPoints", n);
        program.uniform1f("exaggeration", exaggeration);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_POSITIVE_FORCES]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_INTERP_FIELDS]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFF_SUM_Q]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFF_GRADIENTS]);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glDispatchCompute(ceilDiv(n, 128u), 1, 1);
      }

      TOCK_TIMER(TIMR_GRADIENTS);
      ASSERT_GL("GpgpuSneCompute::compute::gradients()");
    }

    // Update embedding
    {
      TICK_TIMER(TIMR_UPDATE);

      // Single computation instead of six values
      float iterMult = (static_cast<double>(iteration) < _params._mom_switching_iter) 
                     ? _params._momentum 
                     : _params._final_momentum;

      auto &program = _programs[PROG_UPDATE];
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("eta", _params._eta);
      program.uniform1f("minGain", _params._minimum_gain);
      program.uniform1f("mult", mult);
      program.uniform1f("iterMult", iterMult);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFF_PREV_GRADIENTS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFF_GAIN]);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      
      TOCK_TIMER(TIMR_UPDATE);
      ASSERT_GL("GpgpuSneCompute::compute::update()");
    }

    // Center embedding
    {
      TICK_TIMER(TIMR_CENTER);

      vec center = _bounds.center();
      vec range = _bounds.range();
      float scaling = 1.f;

      // Do scaling for small embeddings
      if (exaggeration > 1.2f && range.y < 0.1f) {
        scaling = 0.1f / range.y; // TODO Fix for 3D
      }

      auto &program = _programs[PROG_CENTER];
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("scaling", scaling);
      if constexpr (D == 2) {
        program.uniform2f("center", center.x, center.y);
      } else if constexpr (D == 3) {
        program.uniform3f("center", center.x, center.y, center.z);
      }

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFF_POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFF_BOUNDS]);

      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      
      TOCK_TIMER(TIMR_CENTER);
      ASSERT_GL("GpgpuSneCompute::compute::center()");
    }

    POLL_TIMERS()

    if (iteration == _params._iterations - 1) {
      // Update host embedding data
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glGetNamedBufferSubData(_buffers[BUFF_POSITION], 0, n * sizeof(vec), embeddingPtr->getContainer().data());

      // Output timer averages
      utils::secureLog(_logger, "\nGradient descent");
      LOG_TIMER(_logger, TIMR_BOUNDS, "  Bounds")
      LOG_TIMER(_logger, TIMR_SUM_Q, "  Sum Q")
      LOG_TIMER(_logger, TIMR_GRADIENTS, "  Grads")
      LOG_TIMER(_logger, TIMR_UPDATE, "  Update")
      LOG_TIMER(_logger, TIMR_CENTER, "  Center")
      utils::secureLog(_logger, "");
    }

    ASSERT_GL("GpgpuSneCompute::compute()");
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class GpgpuSneCompute<2>;
  template class GpgpuSneCompute<3>;
}