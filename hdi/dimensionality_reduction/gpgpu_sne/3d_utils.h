#pragma once

#include <iostream>
#include <vector>
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/glad/glad.h"

namespace hdi::dr {
  // Internal types
  typedef hdi::data::Embedding<float> embedding_t;
  typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_matrix_t;
  
  // Simple point representation
  // vec3 should get padded to vec4 in std430, so we align to 16 bytes
  struct alignas(16) Point3D {
    float x;
    float y;
    float z;
  };

  // Simple bounds representation
  struct Bounds3D {
    Point3D min;
    Point3D max;

    Point3D range() {
      return Point3D {
        max.x - min.x,
        max.y - min.y,
        max.z - min.z
      };
    }

    Point3D center() {
      return Point3D {
        0.5f * (min.x + max.x),
        0.5f * (min.y + max.y),
        0.5f * (min.z + max.z), 
      };
    }
  };

  // Linearized representation of sparse probability matrix
  struct LinearProbabilityMatrix {
    std::vector<uint32_t> neighbours;
    std::vector<float> probabilities;
    std::vector<int> indices;
  };

  inline LinearProbabilityMatrix linearizeProbabilityMatrix(const embedding_t* embedding, const sparse_matrix_t& P) {
    LinearProbabilityMatrix LP;
    LP.indices.reserve(2 * embedding->numDataPoints());
    LP.neighbours.reserve(500 * embedding->numDataPoints());
    LP.probabilities.reserve(500 * embedding->numDataPoints());

    for (size_t i = 0; i < embedding->numDataPoints(); i++) {
      LP.indices.push_back(static_cast<int>(LP.neighbours.size()));\
      for (const auto& p_ij : P[i]) {
        LP.neighbours.push_back(p_ij.first);
        LP.probabilities.push_back(p_ij.second);
      }
      LP.indices.push_back(static_cast<int>(P[i].size()));
    }
    return LP;
  }
  
  // For aggressive debugging
  inline void glAssert(const std::string& msg) {
    GLenum err; 
    while ((err = glGetError()) != GL_NO_ERROR) {
      std::cerr << "Location: " << msg << ", err: " << err << std::endl;
      exit(0);
    }
  }
};