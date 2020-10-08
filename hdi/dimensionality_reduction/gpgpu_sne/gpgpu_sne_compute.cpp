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

// Magic numbers for the field computation
constexpr bool doAdaptiveResolution = true;
constexpr unsigned minFieldSize = 5;                // 5 is default
constexpr unsigned fixedFieldSize = 40;             // 40 is default
constexpr float pixelRatio = 1.4f;                  // 2.0 in 2d, < 1.35 in 3d seems to work fine
constexpr float boundsPadding = 0.1f;               // Padding to forcedly grow embedding

namespace hdi::dr {
  /**
   * GpgpuSneCompute::GpgpuSneCompute()
   * 
   * Class constructor of GpgpuSneCompute class.
   */
  template <unsigned D>
  GpgpuSneCompute<D>::GpgpuSneCompute()
  : _isInit(false), _logger(nullptr)
  { }
  
  /**
   * GpgpuSneCompute::~GpgpuSneCompute()
   * 
   * Class destructor of GpgpuSneCompute class.
   */
  template <unsigned D>
  GpgpuSneCompute<D>::~GpgpuSneCompute()
  {
    if (_isInit) {
      destr();
    }
  }

  /**
   * GpgpuSneCompute::init()
   * 
   * Function to initialize subclasses and subcomponents of this class. Acquires and initializes
   * buffers and shader programs.
   */
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
        _programs(ProgramType::eBounds).addShader(COMPUTE, _2d::bounds_src);
        _programs(ProgramType::eSumQ).addShader(COMPUTE, _2d::sumq_src);
        _programs(ProgramType::ePositiveForces).addShader(COMPUTE, _2d::positive_forces_src);
        _programs(ProgramType::eGradients).addShader(COMPUTE, _2d::gradients_src);
        _programs(ProgramType::eUpdate).addShader(COMPUTE, _2d::update_src);
        _programs(ProgramType::eCenter).addShader(COMPUTE, _2d::center_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBounds).addShader(COMPUTE, _3d::bounds_src);
        _programs(ProgramType::eSumQ).addShader(COMPUTE, _3d::sumq_src);
        _programs(ProgramType::ePositiveForces).addShader(COMPUTE, _3d::positive_forces_src);
        _programs(ProgramType::eGradients).addShader(COMPUTE, _3d::gradients_src);
        _programs(ProgramType::eUpdate).addShader(COMPUTE, _3d::update_src);
        _programs(ProgramType::eCenter).addShader(COMPUTE, _3d::center_src);
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
      glNamedBufferStorage(_buffers(BufferType::ePosition), n * sizeof(vec), data, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduceAdd), 256 * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eSumQ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSumQReduceAdd), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eInterpFields), n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighbours), LP.neighbours.size() * sizeof(uint32_t), LP.neighbours.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eProbabilities), LP.probabilities.size() * sizeof(float), LP.probabilities.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eIndices), LP.indices.size() * sizeof(int), LP.indices.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::ePositiveForces), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), n * sizeof(vec), ones.data(), 0);

      ASSERT_GL("GpgpuSneCompute::init::buffers()");
      utils::secureLogValue(_logger, "   GpgpuSne", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    // Initialize field computation subcomponent
    if constexpr (D == 2) {
      _field2dCompute.setLogger(_logger);
      _field2dCompute.init(params, _buffers(BufferType::ePosition), _buffers(BufferType::eBounds), embedding->numDataPoints());
    } else if constexpr (D == 3) {
      _field3dCompute.setLogger(_logger);
      _field3dCompute.init(params, _buffers(BufferType::ePosition), _buffers(BufferType::eBounds), embedding->numDataPoints());
    }

    // Initialize embedding renderer subcomponent
    _embeddingRenderer.init(
      embedding->numDataPoints(), 
      _buffers(BufferType::ePosition), 
      _buffers(BufferType::eBounds)
    );

    INIT_TIMERS();
    ASSERT_GL("GpgpuSneCompute::init()");
    _isInit = true;
    std::cerr << "GpgpuSneCompute::init()" << std::endl;
  }

  /**
   * GpgpuSneCompute::destr()
   * 
   * Function to clean up subclasses and subcomponents of this class. Destroys acquired buffers and 
   * shader programs.
   */
  template <unsigned D>
  void GpgpuSneCompute<D>::destr()
  {
    // Destroy field computation subcomponent
    if constexpr (D == 2) {
      _field2dCompute.destr();
    } else if constexpr (D == 3) {
      _field3dCompute.destr();
    }

    // Destroy renderer subcomponent
    _embeddingRenderer.destr();

    // Destroy own components
    for (auto &program : _programs) {
      program.destroy();
    }
    glDeleteBuffers(_buffers.size(), _buffers.data());
    DSTR_TIMERS();
    ASSERT_GL("GpgpuSneCompute::destr()");
    _isInit = false;
  }

  /**
   * GpgpuSneCompute::compute()
   * 
   * Perform one step of the tSNE minimization for the provided embedding.
   *
   */
  template <unsigned D>
  void GpgpuSneCompute<D>::compute(Embedding* embedding,
                                   float exaggeration,
                                   unsigned iteration,
                                   float mult)
  {
    ASSERT_GL("GpgpuSneCompute::compute::begin()");

    const unsigned int n = embedding->numDataPoints();

    // Compute embedding bounds
    {
      TICK_TIMER(TIMR_BOUNDS);

      auto &program = _programs(ProgramType::eBounds);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("padding", boundsPadding);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBoundsReduceAdd));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eBounds));

      program.uniform1ui("iter", 0u);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("GpgpuSneCompute::compute::bounds()");
      TOCK_TIMER(TIMR_BOUNDS);
    }
    
    // Compute field approximation, delegated to subclass
    {
      // Copy bounds back to host (hey look a halt)
      glGetNamedBufferSubData(_buffers(BufferType::eBounds), 0, sizeof(Bounds),  &_bounds);

      // Determine field texture dimensions
      vec range = _bounds.range();
      uvec dims = doAdaptiveResolution
                ? dr::max(uvec(range * pixelRatio), uvec(minFieldSize))
                : uvec(fixedFieldSize);

      // Delegate to subclass depending on dimension of embedding
      if constexpr (D == 2) {
        _field2dCompute.compute(
          dims, iteration, n,
          _buffers(BufferType::ePosition),
          _buffers(BufferType::eBounds),
          _buffers(BufferType::eInterpFields)
        );
      } else if constexpr (D == 3) {
        _field3dCompute.compute(
          dims, iteration, n,
          _buffers(BufferType::ePosition),
          _buffers(BufferType::eBounds),
          _buffers(BufferType::eInterpFields)
        );
      }
    }

    // Calculate sum over Q. Partially called Z or approx(Z) in notation.
    {
      TICK_TIMER(TIMR_SUM_Q);

      auto &program = _programs(ProgramType::eSumQ);
      program.bind();
      program.uniform1ui("nPoints", n);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eInterpFields));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eSumQReduceAdd));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSumQ));

      program.uniform1ui("iter", 0u);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("GpgpuSneCompute::compute::sum_q()");
      TOCK_TIMER(TIMR_SUM_Q);
    }

    // Compute positive/attractive forces
    // This is pretty much a sparse matrix multiplication, and can be really expensive when 
    // perplexity value becomes large. Technically O(kN) for N points and k neighbours, but
    // has horrible disjointed memory access patterns, so for large k it's kinda slow.
    {
      TICK_TIMER(TIMR_F_ATTR);

      auto &program = _programs(ProgramType::ePositiveForces);
      program.bind();
      program.uniform1ui("nPos", n);
      program.uniform1f("invPos", 1.f / static_cast<float>(n));

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNeighbours));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eProbabilities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eIndices));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePositiveForces));

      glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("GpgpuSneCompute::compute::pos_forces()");
      TOCK_TIMER(TIMR_F_ATTR);
    }

    // Compute resulting gradient
    {
      TICK_TIMER(TIMR_GRADIENTS);

      auto &program = _programs(ProgramType::eGradients);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("exaggeration", exaggeration);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePositiveForces));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eInterpFields));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSumQ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      ASSERT_GL("GpgpuSneCompute::compute::gradients()");
      TOCK_TIMER(TIMR_GRADIENTS);
    }

    // Update embedding
    {
      TICK_TIMER(TIMR_UPDATE);

      // Precompute instead of doing it in shader N times
      float iterMult = (static_cast<double>(iteration) < _params._mom_switching_iter) 
                     ? _params._momentum 
                     : _params._final_momentum;

      auto &program = _programs(ProgramType::eUpdate);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("eta", _params._eta);
      program.uniform1f("minGain", _params._minimum_gain);
      program.uniform1f("mult", mult);
      program.uniform1f("iterMult", iterMult);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGain));

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      ASSERT_GL("GpgpuSneCompute::compute::update()");
      TOCK_TIMER(TIMR_UPDATE);
    }

    // Center embedding
    {
      TICK_TIMER(TIMR_CENTER);

      vec center = _bounds.center();
      vec range = _bounds.range();
      float scaling = 1.f;

      // Do scaling for small embeddings
      if (exaggeration > 1.2f && range.y < 0.1f) {
        scaling = 0.1f / range.y; // TODO Fix for 3D?
      }

      auto &program = _programs(ProgramType::eCenter);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("scaling", scaling);
      if constexpr (D == 2) {
        program.uniform2f("center", center.x, center.y);
      } else if constexpr (D == 3) {
        program.uniform3f("center", center.x, center.y, center.z);
      }

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBounds));

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      ASSERT_GL("GpgpuSneCompute::compute::center()");
      TOCK_TIMER(TIMR_CENTER);
    }

    POLL_TIMERS();

    if (iteration == _params._iterations - 1) {
      // Update host embedding data
      glGetNamedBufferSubData(_buffers(BufferType::ePosition), 
        0, n * sizeof(vec), embedding->getContainer().data());

      // Output timer averages
#ifdef GL_TIMERS_ENABLED
      utils::secureLog(_logger, "\nGradient descent");
#endif
      LOG_TIMER(_logger, TIMR_BOUNDS,     "  Bounds")
      LOG_TIMER(_logger, TIMR_SUM_Q,      "  Sum Q")
      LOG_TIMER(_logger, TIMR_F_ATTR,     "  Attr")
      LOG_TIMER(_logger, TIMR_GRADIENTS,  "  Grads")
      LOG_TIMER(_logger, TIMR_UPDATE,     "  Update")
      LOG_TIMER(_logger, TIMR_CENTER,     "  Center")
#ifdef GL_TIMERS_ENABLED
      utils::secureLog(_logger, "");
#endif
    }

    ASSERT_GL("GpgpuSneCompute::compute::end()");
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class GpgpuSneCompute<2>;
  template class GpgpuSneCompute<3>;
}