#ifndef __APPLE__

#include "3d_gpgpu_sne_compute.h"
#include "3d_compute_shaders.h"
#include <limits>

namespace {
  // Magic numbers
  constexpr bool adaptiveResolution = true;
  const unsigned int fixedFieldSize = 40;
  const unsigned int minFieldSize = 5;
  const float pixelRatio = 2.0f;
  const float functionSupport = 6.5f;

  enum BufferType {
    BUFFER_POSITION,
    BUFFER_INTERP_FIELDS,
    BUFFER_SUM_Q,
    BUFFER_NEIGHBOUR,
    BUFFER_PROBABILITIES,
    BUFFER_INDEX,
    BUFFER_GRADIENTS,
    BUFFER_PREV_GRADIENTS,
    BUFFER_GAIN,
    BUFFER_BOUNDS
  };

  enum ProgramType {
    PROGRAM_INTERP_FIELDS,
    PROGRAM_SUM_Q,
    PROGRAM_FORCES,
    PROGRAM_UPDATE,
    PROGRAM_BOUNDS,
    PROGRAM_CENTERING
  };

  typedef hdi::dr::embedding_t embedding_t;
  typedef hdi::dr::Bounds3D Bounds3D;

  Bounds3D computeEmbeddingBounds(const embedding_t* embedding, float padding) {
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
  // Default constr.
  Gpgpu3dSneCompute::Gpgpu3dSneCompute() 
  : _initialized(false), 
    _adaptive_resolution(adaptiveResolution), 
    _resolution_scaling(pixelRatio) {
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
    glClearColor(0, 0, 0, 0);
    const int n = embedding->numDataPoints();
    _params = params;
    _bounds = computeEmbeddingBounds(embedding, 0.f);
    _fieldComputation.initialize(n);

    // Create shader programs 
    try {
      for (auto& program : _programs) {
        program.create();
      }

      // Attach shaders
      _programs[PROGRAM_INTERP_FIELDS].addShader(COMPUTE, interpolation_src);
      _programs[PROGRAM_SUM_Q].addShader(COMPUTE, sumq_src);
      _programs[PROGRAM_FORCES].addShader(COMPUTE, forces_src);
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
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), nullptr, GL_STREAM_READ);
      
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_NEIGHBOUR]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.neighbours.size() * sizeof(uint32_t), LP.neighbours.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_PROBABILITIES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.probabilities.size() * sizeof(float), LP.probabilities.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_INDEX]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.indices.size() * sizeof(int), LP.indices.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_PREV_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), zeroes.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_GAIN]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), ones.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(Point3D), ones.data(), GL_STREAM_READ);
    }

    glAssert("Initialized buffers");
    _initialized = true;
  }

  void Gpgpu3dSneCompute::clean() {
    if (_initialized) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      for (auto& program : _programs) {
        program.destroy();
      }
      _fieldComputation.clean();
      _initialized = false;
    }
  }

  void Gpgpu3dSneCompute::compute(embedding_t* embedding, 
                                  float exaggeration, 
                                  float iteration, 
                                  float mult) {
    unsigned int n = embedding->numDataPoints();

    // Copy data over to device
    if (iteration < 0.5f) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITION]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
    }

    // Compute bounds of the embedding and add 10% border around it
    computeBounds(n, 0.05f);
    glAssert("Computed bounds");

    // Capture padded bounds from gpu
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_BOUNDS]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Bounds3D), &_bounds);

    // Determine fields texture dimension (this has a rather significant impact in 3d)
    Point3D range = _bounds.range();
    float aspectY = range.y / range.x;
    float aspectZ = range.z / range.x;
    uint32_t w = _adaptive_resolution 
      ? std::max((unsigned int)(range.x * _resolution_scaling), minFieldSize) 
      : fixedFieldSize;
    uint32_t h = _adaptive_resolution 
      ? std::max((unsigned int)(range.y * _resolution_scaling), minFieldSize) 
      : (int) (fixedFieldSize * aspectY);
    uint32_t d = _adaptive_resolution 
      ? std::max((unsigned int)(range.z * _resolution_scaling), minFieldSize) 
      : (int) (fixedFieldSize * aspectZ);

    // Compute fields texture
    _fieldComputation.compute(w, h,  d, 
                              functionSupport, n, 
                              _buffers[BUFFER_POSITION], _buffers[BUFFER_BOUNDS], 
                              _bounds.min.x, _bounds.min.y, _bounds.min.z, 
                              _bounds.max.x, _bounds.max.y, _bounds.max.z);

    // Calculate normalization sum and interpolate fields to sample points
    float sum_Q = 0;
    interpolateFields(n, &sum_Q);
    sumQ(n, &sum_Q);
    if (sum_Q == 0) {
      std::cout << "SUM_Q was 0, bring out your debugger" << std::endl;
      return;
    }

    // Compute gradient from forces
    computeGradients(n, sum_Q, exaggeration);

    // Update embedding
    updatePoints(n, iteration, mult);
    updateEmbedding(n, exaggeration, iteration);
    
    // Update host embedding data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_POSITION]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
  }

  void Gpgpu3dSneCompute::computeBounds(unsigned n, float padding) {
    auto& program = _programs[PROGRAM_BOUNDS];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_BOUNDS]);

    // Set uniforms for this shader
    program.uniform1f("padding", padding);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);

    program.release();
  }

  void Gpgpu3dSneCompute::interpolateFields(unsigned n, float* sum_Q) {
    auto& program = _programs[PROGRAM_INTERP_FIELDS];
    program.bind();
    
    // Bind the 3d xyzw fields texture for this shader
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _fieldComputation.getFieldTexture());

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_BOUNDS]);

    // Bind uniforms for this shader
    program.uniform1i("fields", 0);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    program.release();
  }

  void Gpgpu3dSneCompute::sumQ(unsigned n, float *sum_Q) {
    auto& program = _programs[PROGRAM_SUM_Q];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_SUM_Q]);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);

    // Copy SUMQ back to host
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_SUM_Q]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), sum_Q);

    program.release();
  }

  void Gpgpu3dSneCompute::computeGradients(unsigned n, float sum_Q, float exaggeration) {
    auto& program = _programs[PROGRAM_FORCES];
    program.bind();

    // Bindd buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_NEIGHBOUR]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BUFFER_PROBABILITIES]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[BUFFER_INDEX]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[BUFFER_INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers[BUFFER_GRADIENTS]);

    // Bind uniforms for this shader
    program.uniform1f("exaggeration", exaggeration);
    program.uniform1f("sum_Q", sum_Q);

    // Run shader
    glDispatchCompute(n, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    program.release();
  }

  void Gpgpu3dSneCompute::updatePoints(unsigned n, float iteration, float mult) {
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
      (iteration < _params._mom_switching_iter) 
      ? _params._momentum 
      : _params._final_momentum;
    program.uniform1f("iter_mult", iter_mult);

    // Run shader
    glDispatchCompute((n / GLSL_LOCAL_SIZE) + 1, 1, 1);

    program.release();
  }

  void Gpgpu3dSneCompute::updateEmbedding(unsigned n, float exaggeration, float iteration) {
    auto& program = _programs[PROGRAM_CENTERING];
    program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_BOUNDS]);
    
    // Bind uniforms for this shader, depending on exaggeration factor
    if (exaggeration > 1.2) {
      program.uniform1i("scale", 1);
      program.uniform1f("diameter", 0.1f);
    } else {
      program.uniform1i("scale", 0);
    }

    // Run shader
    glDispatchCompute((n / GLSL_LOCAL_SIZE) + 1, 1, 1);

    program.release();
  }
}

#endif