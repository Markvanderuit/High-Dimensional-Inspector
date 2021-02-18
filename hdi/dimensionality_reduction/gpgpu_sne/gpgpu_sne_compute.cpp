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
#include <random>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute_shaders_2d.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_sne_compute_shaders_3d.h"

namespace hdi::dr {
  /**
   * GpgpuSneCompute::GpgpuSneCompute()
   * 
   * Class constructor of GpgpuSneCompute class.
   */
  template <unsigned D>
  GpgpuSneCompute<D>::GpgpuSneCompute()
  : _isInit(false),
    _logger(nullptr)
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
  void GpgpuSneCompute<D>::init(const GpgpuHdCompute::Buffers distribution, const TsneParameters &params)
  {
    hdi::utils::secureLog(_logger, "Initializing minimization"); 
    _distribution = distribution;
    _params = params;

    // Generate embedding data with randomly placed points
    const uint n = _params.n;
    std::vector<vec> embedding(n, vec(0.f));
    utils::secureLog(_logger, "Initializing embedding...");
    {
      // Seed bad rng
      std::srand(_params.seed);

      // Generate n random vectors
      for (uint i = 0; i < n; ++i) {
        vec v;
        float r;

        do {
          r = 0.f;
          for (uint j = 0; j < D; ++j) {
            v[j] = 2.f * (static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.f)) - 1.f;
          }
          r = dr::dot(v, v);
        } while (r > 1.f || r == 0.f);

        r = std::sqrt(-2.f * std::log(r) / r);
        embedding[i] = v * r * _params.rngRange;
      }
    }

    // Initialize shader program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      // Set shaders depending on required dimension
      if constexpr (D == 2) {
        _programs(ProgramType::eBounds).addShader(COMPUTE, _2d::bounds_src);
        _programs(ProgramType::eSumQ).addShader(COMPUTE, _2d::sumq_src);
        _programs(ProgramType::eAttr).addShader(COMPUTE, _2d::positive_forces_src);
        _programs(ProgramType::eGradients).addShader(COMPUTE, _2d::gradients_src);
        _programs(ProgramType::eUpdate).addShader(COMPUTE, _2d::update_src);
        _programs(ProgramType::eCenter).addShader(COMPUTE, _2d::center_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBounds).addShader(COMPUTE, _3d::bounds_src);
        _programs(ProgramType::eSumQ).addShader(COMPUTE, _3d::sumq_src);
        _programs(ProgramType::eAttr).addShader(COMPUTE, _3d::positive_forces_src);
        _programs(ProgramType::eGradients).addShader(COMPUTE, _3d::gradients_src);
        _programs(ProgramType::eUpdate).addShader(COMPUTE, _3d::update_src);
        _programs(ProgramType::eCenter).addShader(COMPUTE, _3d::center_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
      glAssert("GpgpuSneCompute::init::programs()");
    }
    
    // Initialize buffer objects
    {
      const std::vector<vec> zeroes(n, vec(0));
      const std::vector<vec> ones(n, vec(1));

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::ePosition), n * sizeof(vec), embedding.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduceAdd), 256 * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eSumQ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSumQReduceAdd), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eInterpFields), n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttr), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), n * sizeof(vec), ones.data(), 0);
      glAssert("GpgpuSneCompute::init::buffers()");
      utils::secureLogValue(_logger, "   GpgpuSne", std::to_string(bufferSize(_buffers) / 1'000'000) + "mb");
    }

    // Initialize embedding renderer subcomponent
    _embeddingRenderer.init(
      n, 
      _buffers(BufferType::ePosition), 
      _buffers(BufferType::eBounds)
    );

    // Initialize field computation subcomponent
    if constexpr (D == 2) {
      _field2dCompute.setLogger(_logger);
      _field2dCompute.init(params, _buffers(BufferType::ePosition), _buffers(BufferType::eBounds), n);
    } else if constexpr (D == 3) {
      _field3dCompute.setLogger(_logger);
      _field3dCompute.init(params, _buffers(BufferType::ePosition), _buffers(BufferType::eBounds), n);
    }

    glCreateTimers(_timers.size(), _timers.data());
    glAssert("GpgpuSneCompute::init()");
    _isInit = true;
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
    glDeleteTimers(_timers.size(), _timers.data());
    glAssert("GpgpuSneCompute::destr()");
    _isInit = false;
  }

  /**
   * GpgpuSneCompute::compute()
   * 
   * Perform one step of the tSNE minimization for the provided embedding.
   */
  template <unsigned D>
  void GpgpuSneCompute<D>::compute(unsigned iteration, float mult)
  {
    const uint n = _params.n;
    
    // Compute exaggeration factor
    float exaggeration = 1.0f;
    if (iteration <= _params.removeExaggerationIter) {
      exaggeration = _params.exaggerationFactor;
    } else if (iteration <= _params.removeExaggerationIter + _params.exponentialDecayIter) {
      float decay = 1.0f
                  - static_cast<float>(iteration - _params.removeExaggerationIter)
                  / static_cast<float>(_params.exponentialDecayIter);
      exaggeration = 1.0f + (_params.exaggerationFactor - 1.0f) * decay;
    }

    // Compute embedding bounds
    {
      auto &timer = _timers(TimerType::eBounds);
      timer.tick();

      auto &program = _programs(ProgramType::eBounds);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("padding", 0.1f);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBoundsReduceAdd));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eBounds));

      program.uniform1ui("iter", 0u);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuSneCompute::compute::bounds()");
      timer.tock();
    }
    
    // Compute field approximation, delegated to subclass
    {
      // Copy bounds back to host (hey look a halt)
      // FIXME Maybe speed this up by copying between buffers earlier on
      glGetNamedBufferSubData(_buffers(BufferType::eBounds), 0, sizeof(Bounds),  &_bounds);

      // Determine field texture dimensions
#ifdef FIELD_DO_ADAPTIVE_RESOLUTION
      const vec range = _bounds.range();
      const float ratio = D == 2 ?  _params.texture2dScaling : _params.texture3dScaling;
      uvec dims = dr::max(uvec(range * ratio), uvec(FIELD_MIN_SIZE));
      dims = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(dims.x)) / glm::log(2.f))));
#else
      const uvec dims = uvec(FIELD_FIXED_SIZE);
#endif

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
      auto &timer = _timers(TimerType::eSumQ);
      timer.tick();

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

      glAssert("GpgpuSneCompute::compute::sum_q()");
      timer.tock();
    }

    // Compute positive/attractive forces
    // This is pretty much a sparse matrix multiplication, and can be really expensive when 
    // perplexity value becomes large. Technically O(kN) for N points and k neighbours, but
    // has horrible disjointed memory access patterns, so for large k it's kinda slow.
    {
      auto &timer = _timers(TimerType::eAttr);
      timer.tick();

      auto &program = _programs(ProgramType::eAttr);
      program.bind();
      program.uniform1ui("nPos", n);
      program.uniform1f("invPos", 1.f / static_cast<float>(n));

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _distribution.layoutBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _distribution.neighboursBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _distribution.similaritiesBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttr));

      glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1); // divide by subgroup size
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuSneCompute::compute::pos_forces()");
      timer.tock();
    }

    // Compute resulting gradient
    {
      auto &timer = _timers(TimerType::eGrads);
      timer.tick();

      auto &program = _programs(ProgramType::eGradients);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("exaggeration", exaggeration);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttr));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eInterpFields));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eSumQ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuSneCompute::compute::gradients()");
      timer.tock();
    }

    // Update embedding
    {
      auto &timer = _timers(TimerType::eUpdate);
      timer.tick();

      // Precompute instead of doing it in shader N times
      float iterMult = (static_cast<double>(iteration) < _params.momentumSwitchIter) 
                     ? _params.momentum 
                     : _params.finalMomentum;

      auto &program = _programs(ProgramType::eUpdate);
      program.bind();
      program.uniform1ui("nPoints", n);
      program.uniform1f("eta", _params.eta);
      program.uniform1f("minGain", _params.minimumGain);
      program.uniform1f("mult", mult);
      program.uniform1f("iterMult", iterMult);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePosition));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGain));

      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      glAssert("GpgpuSneCompute::compute::update()");
      timer.tock();
    }

    // Center embedding
    {
      auto &timer = _timers(TimerType::eCenter);
      timer.tick();

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
      
      glAssert("GpgpuSneCompute::compute::center()");
      timer.tock();
    }

    glPollTimers(_timers.size(), _timers.data());
  }

  template <unsigned D>
  std::vector<glm::vec<D, float>> GpgpuSneCompute<D>::getEmbedding() const {
    if (!_isInit) {
      throw std::runtime_error("GpgpuSneCompute::getEmbedding() called while GpgpuSneCompute was not initialized");
    }
    std::vector<vec> buffer(_params.n);
    glGetNamedBufferSubData(_buffers(BufferType::ePosition), 0, buffer.size() * sizeof(vec), buffer.data());
    return std::vector<glm::vec<D, float>>(std::begin(buffer), std::end(buffer));
  }

  template <unsigned D>
  void GpgpuSneCompute<D>::logTimerAverage() const {
    if constexpr (D == 2) {
      _field2dCompute.logTimerAverage();
    } else if constexpr (D == 3) {
      _field3dCompute.logTimerAverage();
    }

    utils::secureLog(_logger, "\nGradient descent");
    LOG_TIMER(_logger, TimerType::eBounds,     "  Bounds");
    LOG_TIMER(_logger, TimerType::eSumQ,      "  Sum Q");
    LOG_TIMER(_logger, TimerType::eAttr,     "  Attr");
    LOG_TIMER(_logger, TimerType::eGrads,  "  Grads");
    LOG_TIMER(_logger, TimerType::eUpdate,     "  Update");
    LOG_TIMER(_logger, TimerType::eCenter,     "  Center");
  }
  
  // Explicit template instantiations for 2 and 3 dimensions
  template class GpgpuSneCompute<2>;
  template class GpgpuSneCompute<3>;
}