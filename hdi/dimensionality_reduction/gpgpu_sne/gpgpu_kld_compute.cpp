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

#include <iostream>
#include <stdexcept>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_kld_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_kld_compute_shaders.h"

namespace hdi::dr {
  template <unsigned D>
  GpgpuKldCompute<D>::GpgpuKldCompute()
  : _logger(nullptr) { }

  template <unsigned D>
  float GpgpuKldCompute<D>::compute(const typename GpgpuSneCompute<D>::Buffers embedding,
                                    const GpgpuHdCompute::Buffers distribution,
                                    const TsneParameters &params) {
    const unsigned n = params.n;

    // Create shader program objects
    {
      for (auto &program : _programs) {
        program.create();
      }

      if constexpr (D == 2) {
        _programs(ProgramType::eQij).addShader(COMPUTE, _2d::qij_src);
        _programs(ProgramType::eReduce).addShader(COMPUTE, _2d::reduce_src);
        _programs(ProgramType::eKl).addShader(COMPUTE, _2d::kl_src);
      } else if constexpr (D == 3) {
        _programs(ProgramType::eQij).addShader(COMPUTE, _3d::qij_src);
        _programs(ProgramType::eReduce).addShader(COMPUTE, _2d::reduce_src);
        _programs(ProgramType::eKl).addShader(COMPUTE, _3d::kl_src);
      }

      for (auto &program : _programs) {
        program.build();
      }
    }

    // Create shader storage buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eQij), n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eKlc), n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eReduceIntermediate), 256 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eReduceFinal), sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

      glAssert("GpgpuKldCompute::compute::buffers()");
    }

    glCreateTimers(_timers.size(), _timers.data());

    // Compute Q_ij over all i without approximation, therefore in O(n^2) time
    {
      auto &timer = _timers(TimerType::eQij);
      timer.tick();

      auto &program = _programs(ProgramType::eQij);
      program.bind();
      program.uniform1ui("nPoints", n);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, embedding.positionsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eQij));
      
      const uint step = 1024;
      const uint end = n;
      for (int begin = 0; begin < end; begin += step) {
        program.uniform1ui("begin", begin);
        glDispatchCompute(std::min(step, end - begin), 1, 1);
      }
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuKldCompute::compute::q_ij()");
      timer.tock();
    }

    // Compute sum over Q_ij, eg. Z in paper, through parallel reduction.
    {
      auto &timer = _timers(TimerType::eReduceQij);
      timer.tick();

      auto &program = _programs(ProgramType::eReduce);
      program.bind();
      program.uniform1ui("nPoints", n);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eQij));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduceIntermediate));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eReduceFinal));

      program.uniform1ui("iter", 0u);
      glDispatchCompute(256, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuKldCompute::compute::reduce0()");
      timer.tock();
    }

    // Compute KL-divergence inner sum
    {
      auto &timer = _timers(TimerType::eKlc);
      timer.tick();

      auto &program = _programs(ProgramType::eKl);
      program.bind();
      program.uniform1ui("nPoints", n);
      
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, embedding.positionsBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduceFinal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, distribution.layoutBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, distribution.neighboursBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, distribution.similaritiesBuffer);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eKlc));

      const uint step = 1024;
      const uint end = n;
      for (int begin = 0; begin < end; begin += step) {
        program.uniform1ui("begin", begin);
        glDispatchCompute(std::min(step, end - begin), 1, 1);
      }
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      glAssert("GpgpuKldCompute::compute::kl_divergence()");
      timer.tock();
    }

    // Compute sum over computed KL values, through parallel reduction.
    {
      auto &timer = _timers(TimerType::eReduceKlc);
      timer.tick();

      auto &program = _programs(ProgramType::eReduce);
      program.bind();
      program.uniform1ui("nPoints", n);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eKlc));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduceIntermediate));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eReduceFinal));
 
      program.uniform1ui("iter", 0u);
      glDispatchCompute(256, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      program.uniform1ui("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      glAssert("GpgpuKldCompute::compute::reduce1()");
      timer.tock();
    }
    
    // Query timers
    {
      // glPollTimers(_timers.size(), _timers.data());
      // LOG_TIMER(_logger, TimerType::eQij, "  Qij compute")
      // LOG_TIMER(_logger, TimerType::eReduceQij, "  Qij reduce")
      // LOG_TIMER(_logger, TimerType::eKlc, "  KLC compute")
      // LOG_TIMER(_logger, TimerType::eReduceKlc, "  KLD reduce")
      // utils::secureLog(_logger, "");
      glDeleteTimers(_timers.size(), _timers.data());
    }

    // Read computed value back from dynamic buffer
    float kld = 0.0;
    glGetNamedBufferSubData(_buffers(BufferType::eReduceFinal), 0, sizeof(float), &kld);

    // Destroy opengl objects
    {
      for (auto &program : _programs) {
        program.destroy();
      }
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
    
    glAssert("GpgpuKldCompute::compute()");

    return kld;
  }

  // Explicit template instantiations for 2 and 3 dimensions
  template class GpgpuKldCompute<2>;
  template class GpgpuKldCompute<3>;
} // namespace hdi::dr