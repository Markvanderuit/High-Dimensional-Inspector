#ifndef __APPLE__

#include "3d_gpgpu_sne_compute.h"
#include "3d_compute_shaders.h"
#include <limits>

namespace {
  void glAssert(const std::string& msg) {
    GLenum err; 
    while ((err = glGetError()) != GL_NO_ERROR) {
      printf("Location: %s, err: %i\n", msg.c_str(), err);
    }
  }

  // Magic numbers
  const unsigned int fixedFieldSize = 40;
  const unsigned int minFieldSize = 5;
  const float pixelRatio = 2;
  const float functionSupport = 6.5;

  enum BufferType {
    POSITION,
    INTERP_FIELDS,
    SUM_Q,
    NEIGHBOUR,
    PROBABILITIES,
    INDEX,
    GRADIENTS,
    PREV_GRADIENTS,
    GAIN,
    BOUNDS
  };
}

namespace hdi::dr {
  typedef Gpgpu3dSneCompute::embedding_t embedding_t;
  typedef Gpgpu3dSneCompute::sparse_matrix_t sparse_matrix_t;
  typedef Gpgpu3dSneCompute::Bounds3D Bounds3D;
  typedef Gpgpu3dSneCompute::LinearProbabilityMatrix LinearProbabilityMatrix;

  // Default constr.
  Gpgpu3dSneCompute::Gpgpu3dSneCompute() 
  : _initialized(false), 
    _adaptive_resolution(true), 
    _resolution_scaling(pixelRatio) {
    // ...
  }

  // hidden function
  Bounds3D computeEmbeddingBounds(const embedding_t* embedding, float padding) {
    const auto& points = embedding->getContainer();

    Bounds3D bounds;

    // Instantiate bounds at infinity
    bounds.min.x = std::numeric_limits<float>::max();
    bounds.max.x = -std::numeric_limits<float>::max();
    bounds.min.y = std::numeric_limits<float>::max();
    bounds.max.y = -std::numeric_limits<float>::max();
    bounds.min.z = std::numeric_limits<float>::max();
    bounds.max.z = -std::numeric_limits<float>::max();

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

  LinearProbabilityMatrix linearizeProbabilityMatrix(const embedding_t* embedding, const sparse_matrix_t& P) {
    LinearProbabilityMatrix LP;
    for (int i = 0; i < embedding->numDataPoints(); i++) {
      LP.indices.push_back(LP.neighbours.size());
      for (const auto& p_ij : P[i]) {
        LP.neighbours.push_back(p_ij.first);
        LP.probabilities.push_back(p_ij.second);
      }
      LP.indices.push_back(P[i].size());
    }
    return LP;
  }

  void Gpgpu3dSneCompute::initialize(const embedding_t* embedding, 
                    const TsneParameters& params, 
                    const sparse_matrix_t& P) {
    glClearColor(0, 0, 0, 0);
    
    _params = params;
    _bounds = computeEmbeddingBounds(embedding, 0.f);
    const int n = embedding->numDataPoints();
    _fieldComputation.initialize(n);

    // Create shader programs 
    try {
      _interpolation_program.create();
      _sumq_program.create();
      _forces_program.create();
      _update_program.create();
      _bounds_program.create();
      _centering_program.create();

      _interpolation_program.addShader(COMPUTE, interpolation_src);
      _sumq_program.addShader(COMPUTE, sumq_src);
      _forces_program.addShader(COMPUTE, forces_src);
      _update_program.addShader(COMPUTE, update_src);
      _bounds_program.addShader(COMPUTE, bounds_src);
      _centering_program.addShader(COMPUTE, centering_src);

      _interpolation_program.build();
      _sumq_program.build();
      _forces_program.build();
      _update_program.build();
      _bounds_program.build();
      _centering_program.build();

      // Constant uniforms can be set beforehand
      _interpolation_program.bind();
      _interpolation_program.uniform1ui("num_points", n);
      _sumq_program.bind();
      _sumq_program.uniform1ui("num_points", n);
      _forces_program.bind();
      _forces_program.uniform1ui("num_points", n);
      _update_program.bind();
      _update_program.uniform1ui("num_points", n);
      _bounds_program.bind();
      _bounds_program.uniform1ui("num_points", n);
      _centering_program.bind();
      _centering_program.uniform1ui("num_points", n);
    } catch (const ShaderLoadingException& e) {
      std::cout << e.what() << std::endl;
    }
    
    // Create and initialize SSBOs
    {
      const LinearProbabilityMatrix LP = linearizeProbabilityMatrix(embedding, P);
      const std::vector<float> zeroes(4 * n, 0);
      const std::vector<float> ones(4 * n, 1);

      glGenBuffers(_buffers.size(), _buffers.data());

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[POSITION]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[INTERP_FIELDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * 4 * sizeof(float), nullptr, GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[SUM_Q]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), nullptr, GL_STREAM_READ);
      
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[NEIGHBOUR]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.neighbours.size() * sizeof(uint32_t), LP.neighbours.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[PROBABILITIES]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.probabilities.size() * sizeof(float), LP.probabilities.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[INDEX]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, LP.indices.size() * sizeof(int), LP.indices.data(), GL_STATIC_DRAW);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), nullptr, GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[PREV_GRADIENTS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), zeroes.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[GAIN]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(Point3D), ones.data(), GL_STREAM_READ);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BOUNDS]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Bounds3D), ones.data(), GL_STREAM_READ);
    }

    glAssert("Initialized buffers");
    _initialized = true;
  }

  void Gpgpu3dSneCompute::clean() {
    glDeleteBuffers(_buffers.size(), _buffers.data());
    _interpolation_program.destroy();
    _sumq_program.destroy();
    _forces_program.destroy();
    _update_program.destroy();
    _bounds_program.destroy();
    _centering_program.destroy();
    _fieldComputation.clean();
  }

  void Gpgpu3dSneCompute::compute(embedding_t* embedding, 
                                  float exaggeration, 
                                  float iteration, 
                                  float mult) {
    unsigned int n = embedding->numDataPoints();

    // Copy data over to device
    if (iteration < 0.5f) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[POSITION]);
      glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
    }

    // Compute bounds of the embedding and add 10% border around it
    computeBounds(n, 0.05f);
    glAssert("Computed bounds");

    // Capture padded bounds from gpu
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BOUNDS]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Bounds3D), &_bounds);
    
    // Determine fields texture dimension (this has a rather significant impact in 3d)
    Point3D range = _bounds.getRange();
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
                              _buffers[POSITION], _buffers[BOUNDS], 
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[POSITION]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(Point3D), embedding->getContainer().data());
  }

  void Gpgpu3dSneCompute::computeBounds(unsigned n, float padding) {
    _bounds_program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BOUNDS]);

    // Set uniforms for this shader
    _bounds_program.uniform1f("padding", padding);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);

    _bounds_program.release();
  }

  void Gpgpu3dSneCompute::interpolateFields(unsigned n, float* sum_Q) {
    _interpolation_program.bind();

    // Bind the 3d xyzw fields texture for this shader
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _fieldComputation.getFieldTexture());

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[BOUNDS]);

    // Bind uniforms for this shader
    _interpolation_program.uniform1i("fields", 0);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    _interpolation_program.release();
  }

  void Gpgpu3dSneCompute::sumQ(unsigned n, float *sum_Q) {
    _sumq_program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[SUM_Q]);

    // Run shader (dimensionality reduction in single group)
    glDispatchCompute(1, 1, 1);

    // Copy SUMQ back to host
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[SUM_Q]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), sum_Q);

    _sumq_program.release();
  }

  void Gpgpu3dSneCompute::computeGradients(unsigned n, float sum_Q, float exaggeration) {
    _forces_program.bind();

    // Bindd buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[NEIGHBOUR]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[PROBABILITIES]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[INDEX]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers[INTERP_FIELDS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers[GRADIENTS]);

    // Bind uniforms for this shader
    _forces_program.uniform1f("exaggeration", exaggeration);
    _forces_program.uniform1f("sum_Q", sum_Q);

    // Run shader
    glDispatchCompute(n, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    _forces_program.release();
  }

  void Gpgpu3dSneCompute::updatePoints(unsigned n, float iteration, float mult) {
    _update_program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[GRADIENTS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers[PREV_GRADIENTS]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers[GAIN]);

    // Bind uniforms for this shader
    _update_program.uniform1f("eta", _params._eta);
    _update_program.uniform1f("minGain", _params._minimum_gain);
    _update_program.uniform1f("mult", mult);

    // Single computation instead of six values
    float iter_mult = 
      (iteration < _params._mom_switching_iter) 
      ? _params._momentum 
      : _params._final_momentum;
    _update_program.uniform1f("iter_mult", iter_mult);

    // Run shader
    glDispatchCompute((n / GLSL_LOCAL_SIZE) + 1, 1, 1);

    _update_program.release();
  }

  void Gpgpu3dSneCompute::updateEmbedding(unsigned n, float exaggeration, float iteration) {
    _centering_program.bind();

    // Bind buffers for this shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[POSITION]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BOUNDS]);
    
    // Bind uniforms for this shader, depending on exaggeration factor
    if (exaggeration > 1.2) {
      _centering_program.uniform1i("scale", 1);
      _centering_program.uniform1f("diameter", 0.1f);
    } else {
      _centering_program.uniform1i("scale", 0);
    }

    // Run shader
    glDispatchCompute((n / GLSL_LOCAL_SIZE) + 1, 1, 1);

    _centering_program.release();
  }
}

#endif