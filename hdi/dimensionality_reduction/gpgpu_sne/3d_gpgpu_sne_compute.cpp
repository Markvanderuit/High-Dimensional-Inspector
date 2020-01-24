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
#include "3d_gpgpu_sne_compute.h"
#include "3d_gpgpu_sne_shaders.h"

// Magic numbers
constexpr bool adaptiveResolution = true;
const unsigned int fixedFieldSize = 40;
const unsigned int fixedDepthSize = 16;
const unsigned int minFieldSize = 5;
const float pixelRatio = 2.0f;
const float functionSupport = 6.5f;

namespace {
  typedef hdi::dr::embedding_t embedding_t;
  typedef hdi::dr::sparse_matrix_t sparse_matrix_t;
  typedef hdi::dr::Point3D Point3D;
  typedef hdi::dr::Bounds3D Bounds3D;

  Bounds3D computeEmbeddingBounds(const embedding_t* embedding, 
                                  float padding) {
    const auto& points = embedding->getContainer();

    // Instantiate bounds at infinity
    Bounds3D bounds;
    bounds.min.x = bounds.min.y = bounds.min.z = std::numeric_limits<float>::max();
    bounds.max.x = bounds.max.y = bounds.max.z = -std::numeric_limits<float>::max();

    // Iterate over bounds to set the largest
    for (int i = 0; i < embedding->numDataPoints(); ++i) {
      const auto x = points[i * 4 + 0];
      const auto y = points[i * 4 + 1];
      const auto z = points[i * 4 + 2];

      bounds.min.x = std::min<float>(x, bounds.min.x);
      bounds.max.x = std::max<float>(x, bounds.max.x);
      bounds.min.y = std::min<float>(y, bounds.min.y);
      bounds.max.y = std::max<float>(y, bounds.max.y);
      bounds.min.z = std::min<float>(z, bounds.min.z);
      bounds.max.z = std::max<float>(z, bounds.max.z);
    }
    
    // Add extra padding if requested
    if (padding != 0.f) {
      const float x_padding = (bounds.max.x - bounds.min.x) * 0.5f * padding;
      const float y_padding = (bounds.max.y - bounds.min.y) * 0.5f * padding;
      const float z_padding = (bounds.max.z - bounds.min.z) * 0.5f * padding;

      bounds.min.x -= x_padding;
      bounds.max.x += x_padding;
      bounds.min.y -= y_padding;
      bounds.max.y += y_padding;
      bounds.min.z -= z_padding;
      bounds.max.z += z_padding;
    }

    return bounds;
  }
}

namespace hdi::dr {
  Gpgpu3dSneCompute::Gpgpu3dSneCompute() 
  : _initialized(false), 
    _adaptive_resolution(adaptiveResolution), 
    _resolution_scaling(pixelRatio),
    _logger(nullptr) {
    // ...
  }

  Gpgpu3dSneCompute::~Gpgpu3dSneCompute() {
    if (_initialized) {
      clean();
    }
  }

  void Gpgpu3dSneCompute::initialize(const embedding_t* embedding, 
                    const TsneParameters& params, 
                    const sparse_matrix_t& P) {
    TIMERS_CREATE()
    glClearColor(0, 0, 0, 0);
    _params = params;
    _bounds = computeEmbeddingBounds(embedding, 0.f);
    const int n = embedding->numDataPoints();
    _fieldComputation.initialize(_params, n);

    // Create shader programs 
    try {
      for (auto& program : _programs) {
        program.create();
      }

      // Attach shaders
      _programs[PROGRAM_SUM_Q].addShader(COMPUTE, sumq_src);
      _programs[PROGRAM_POSITIVE_FORCES].addShader(COMPUTE, positive_forces_src);
      _programs[PROGRAM_GRADIENTS].addShader(COMPUTE, gradients_src);
      _programs[PROGRAM_UPDATE].addShader(COMPUTE, update_src);
      _programs[PROGRAM_BOUNDS].addShader(COMPUTE, bounds_src);
      _programs[PROGRAM_CENTERING].addShader(COMPUTE, centering_src);

      for (auto& program : _programs) {
        program.build();

        // Most progams use this and it remains constant throughout runtime
        program.bind();
        if (program.getUniformLocation("num_points") != -1) {
          program.uniform1ui("num_points", n);
        }
        if (program.getUniformLocation("inv_num_points") != -1) {
          program.uniform1f("inv_num_points", 1.f / static_cast<float>(n));
        }
        program.release();
      }
    } catch (const ErrorMessageException& e) {
      std::cout << e.what() << std::endl;
    }
    
    // Create and initialize SSBOs
    {
      const LinearProbabilityMatrix LP = linearizeProbabilityMatrix(embedding, P);
      const std::vector<float> zeroes(4 * n, 0);
      const std::vector<float> ones(4 * n, 1);
      glGenBuffers(_buffers.size(), _buffers.data());

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITION]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_INTERP_FIELDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * 4 * sizeof(float), nullptr, GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_SUM_Q]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(float), nullptr, GL_STREAM_READ);
      
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_SUM_Q_REDUCE_ADD]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 128 * sizeof(float), nullptr, GL_STREAM_DRAW);
      
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_NEIGHBOUR]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.neighbours.size() * sizeof(uint32_t), LP.neighbours.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_PROBABILITIES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.probabilities.size() * sizeof(float), LP.probabilities.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_INDEX]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.indices.size() * sizeof(int), LP.indices.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITIVE_FORCES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_PREV_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), zeroes.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_GAIN]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), ones.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(Point3D), ones.data(), GL_STREAM_READ);
      
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_BOUNDS_REDUCE_ADD]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * 128 * sizeof(Point3D), ones.data(), GL_STREAM_READ);
    }

    GL_ASSERT("Initialized buffers");
    _initialized = true;
  }

  void Gpgpu3dSneCompute::clean() {
    if (_initialized) {
      TIMERS_DESTROY()
      for (auto& program : _programs) {
        program.destroy();
      }
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _fieldComputation.clean();
      _initialized = false;
    }
  }

  void Gpgpu3dSneCompute::compute(embedding_t* embedding, 
                                  float exaggeration, 
                                  unsigned iteration, 
                                  float mult) {
    const unsigned int n = embedding->numDataPoints();

    // Copy embedding positional data over to device
    if (iteration == 0) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITION]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
    }

    // Compute bounds of the embedding and add 10% border around it
    computeBounds(n, 0.1f);

    // Capture padded bounds from gpu
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_BOUNDS]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(Point3D), &_bounds);

    // Determine fields texture dimension (this has a rather significant impact in 3d)
    Point3D range = _bounds.range();
    
    uint32_t w = _adaptive_resolution 
      ? std::max((unsigned int)(range.x * _resolution_scaling), minFieldSize) 
      : fixedFieldSize;
    uint32_t h = _adaptive_resolution 
      ? std::max((unsigned int)(range.y * _resolution_scaling), minFieldSize) 
      : (int) (fixedFieldSize * (range.y / range.x));
#ifdef USE_DEPTH_FIELD
    uint32_t d = _adaptive_resolution 
      ? std::max((unsigned int)(range.z * _resolution_scaling * 0.25), minFieldSize) 
      : (int) (fixedFieldSize * (range.z / range.x));
#else
    uint32_t d = _adaptive_resolution 
      ? std::max((unsigned int)(range.z * _resolution_scaling), minFieldSize) 
      : (int) (fixedFieldSize * (range.z / range.x));
#endif
    
    // Compute fields texture
    _fieldComputation.compute(w, h,  d, 
                              functionSupport, n, 
                              _buffers[BUFFER_POSITION], 
                              _buffers[BUFFER_BOUNDS], 
                              _buffers[BUFFER_INTERP_FIELDS],
                              _bounds);

    // Calculate normalization sum
    sumQ(n);

#ifdef ASSERT_SUM_Q
    float sum_Q = 0.f;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_SUM_Q]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), &sum_Q);
    if (sum_Q == 0.f) {
      throw std::runtime_error("SUM_Q was 0, bring out thine debuggers");
    }
#endif // ASSERT_SUM_Q

    // Compute gradient from forces
    computeGradients(n, exaggeration);

    // Update embedding
    updatePoints(n, iteration, mult);
    updateEmbedding(n, exaggeration, iteration);
    
    TIMERS_UPDATE()

    // Update host embedding data
    if (iteration >= _params._iterations - 1) {
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITION]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
      
      // Output timer averages
      TIMER_LOG(_logger, TIMER_SUM_Q, "Sum Q")
      TIMER_LOG(_logger, TIMER_GRADIENTS, "Grads")
      TIMER_LOG(_logger, TIMER_UPDATE, "Update")
      TIMER_LOG(_logger, TIMER_BOUNDS, "Bounds")
      TIMER_LOG(_logger, TIMER_CENTERING, "Center")
      utils::secureLog(_logger, "");
    }
  }

  void Gpgpu3dSneCompute::computeBounds(unsigned n, float padding) {
    TIMER_TICK(TIMER_BOUNDS)
    auto& program = _programs[PROGRAM_BOUNDS];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_BOUNDS_REDUCE_ADD]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_BOUNDS]);

    // Set uniforms for this shader
    program.uniform1f("padding", padding);

    // Run shader (dimensionality reduction in single group)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.uniform1ui("iteration", 0u);
    glDispatchCompute(128, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.uniform1ui("iteration", 1u);
    glDispatchCompute(1, 1, 1);

    program.release();
    TIMER_TOCK(TIMER_BOUNDS)
  }

  void Gpgpu3dSneCompute::sumQ(unsigned n) {
    TIMER_TICK(TIMER_SUM_Q)
    auto& program = _programs[PROGRAM_SUM_Q];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_SUM_Q_REDUCE_ADD]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_SUM_Q]);

    // Run shader twice (reduce add performed in two steps)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.uniform1ui("iteration", 0u);
    glDispatchCompute(128, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    program.uniform1ui("iteration", 1u);
    glDispatchCompute(1, 1, 1);

    program.release();
    TIMER_TOCK(TIMER_SUM_Q)
  }

  void Gpgpu3dSneCompute::computeGradients(unsigned n, float exaggeration) {
    TIMER_TICK(TIMER_GRADIENTS)

    // Reduce add positive forces (performance hog, 60K reduce-adds over varying points)
    {
      auto& program = _programs[PROGRAM_POSITIVE_FORCES];
      program.bind();

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_NEIGHBOUR]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_PROBABILITIES]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFFER_INDEX]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[BUFFER_POSITIVE_FORCES]);

      // Run shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(n, 1, 1);

      program.release();
    }

    // Compute gradients
    {    
      auto& program = _programs[PROGRAM_GRADIENTS];
      program.bind();

      // Bind buffers for this shader
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITIVE_FORCES]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_INTERP_FIELDS]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_SUM_Q]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFFER_GRADIENTS]);

      // Bind uniforms for this shader
      program.uniform1f("exaggeration", exaggeration);

      // Run shader
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glDispatchCompute(128, 1, 1);

      program.release();
    }

    TIMER_TOCK(TIMER_GRADIENTS)
  }

  void Gpgpu3dSneCompute::updatePoints(unsigned n, float iteration, float mult) {
    TIMER_TICK(TIMER_UPDATE)
    auto& program = _programs[PROGRAM_UPDATE];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_GRADIENTS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_PREV_GRADIENTS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFFER_GAIN]);
    
    // Bind uniforms for this shader
    program.uniform1f("eta", _params._eta);
    program.uniform1f("minGain", _params._minimum_gain);
    program.uniform1f("mult", mult);

    // Single computation instead of six values
    float iter_mult = 
      (static_cast<double>(iteration) < _params._mom_switching_iter) 
      ? _params._momentum 
      : _params._final_momentum;
    program.uniform1f("iter_mult", iter_mult);

    // Run shader
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDispatchCompute(128, 1, 1);

    program.release();
    TIMER_TOCK(TIMER_UPDATE)
  }

  void Gpgpu3dSneCompute::updateEmbedding(unsigned n, float exaggeration, float iteration) {
    TIMER_TICK(TIMER_CENTERING)
    auto& program = _programs[PROGRAM_CENTERING];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_BOUNDS]);
    
    // Set uniforms for this shader
    Point3D center = _bounds.center();
    Point3D range = _bounds.range();
    if (exaggeration > 1.2f && range.z < 0.1f) {
      // Enable scaling for small embeddings
      program.uniform1f("scaling", 0.1f / range.z);
    } else {
      program.uniform1f("scaling", 1.f);
    }
    program.uniform3f("center", center.x, center.y, center.z);

    // Run shader
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDispatchCompute(128, 1, 1);

    program.release();
    TIMER_TOCK(TIMER_CENTERING)
  }
}