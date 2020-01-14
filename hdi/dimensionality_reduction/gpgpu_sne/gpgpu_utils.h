#pragma once

#include <iostream>
#include <vector>
#include "hdi/data/embedding.h"
#include "hdi/data/map_mem_eff.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/glad/glad.h"

// Toggle to globally enable/disable gl query timers
#define UTILS_QUERY_TIMERS 
#ifdef UTILS_QUERY_TIMERS
#include <array>
#include <deque>
#include <numeric>
#include <stdexcept>
#endif

namespace hdi::dr {
  // Internal types
  typedef hdi::data::Embedding<float> embedding_t;
  typedef std::vector<hdi::data::MapMemEff<uint32_t, float>> sparse_matrix_t;

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

#ifdef UTILS_QUERY_TIMERS
  // GL timer query class
  class glTimerQuery {
  public:
    glTimerQuery()
    : _initialized(false) { }

    ~glTimerQuery() {
      if (_initialized) {
        destroy();
      }
    }
    
    void create() {
      if (!_initialized) {
        glGenQueries(1, &_handle);
        _values.fill(0);
        _iteration = 0;
        _initialized = true;
      }
    }

    void destroy() {
      if (_initialized) {
        glDeleteQueries(1, &_handle);
        _values.fill(0);
        _iteration = 0;
        _initialized = false;
      }
    }

    void tick() {
      assertInitialized();
      glBeginQuery(GL_TIME_ELAPSED, _handle);
    }

    void tock() {
      assertInitialized();
      glEndQuery(GL_TIME_ELAPSED);
    }

    bool poll() {
      assertInitialized();
      int i = 0;
      glGetQueryObjectiv(_handle, GL_QUERY_RESULT_AVAILABLE, &i);
      if (!i) {
        return GL_FALSE;
      }
      
      glGetQueryObjecti64v(_handle, GL_QUERY_RESULT, &_values[TIMER_LAST_QUERY]);
      _values[TIMER_TOTAL] += _values[TIMER_LAST_QUERY];
      _values[TIMER_AVERAGE] 
        = _values[TIMER_AVERAGE] 
        + (_values[TIMER_LAST_QUERY] - _values[TIMER_AVERAGE]) 
        / (++_iteration);

      return GL_TRUE;
    }

    double averageMillis() const {
      return static_cast<double>(_values[TIMER_AVERAGE]) / 1000000.0;
    }

    double totalMillis() const {
      return static_cast<double>(_values[TIMER_TOTAL]) / 1000000.0;
    }

  private:
    enum ValueType {
      TIMER_LAST_QUERY,
      TIMER_AVERAGE,
      TIMER_TOTAL,

      // Static enum length
      ValueTypeLength 
    };

    bool _initialized;
    GLuint _handle;
    unsigned _iteration;
    std::array<GLint64, ValueTypeLength> _values;

    void assertInitialized() {
      if (!_initialized) {
        throw std::runtime_error("glTimerQuery not initialized");
      }
    }
  };

  #define TIMERS_DECLARE(...) \
    enum glTimerQueryType { \
      __VA_ARGS__,\
      glTimerQueryTypeLength \
    }; \
    std::array<glTimerQuery, glTimerQueryTypeLength> glTimerQueries;

  #define TIMERS_CREATE() \
    for (auto& t : glTimerQueries) t.create();
  
  #define TIMERS_DESTROY() \
    for (auto& t : glTimerQueries) t.destroy();

  #define TIMERS_UPDATE() \
    { \
      std::deque<int> types(glTimerQueryTypeLength); \
      std::iota(types.begin(), types.end(), 0); \
      while (!types.empty()) { \
        const int t = types.front(); \
        types.pop_front(); \
        if (!glTimerQueries[t].poll()) { \
          types.push_back(t); \
        } \
      } \
    }
  
  #define TIMER_TICK(name) \
    glTimerQueries[name].tick();

  #define TIMER_TOCK(name) \
    glTimerQueries[name].tock();

  #define TIMER_LOG(logger, name, str) \
    utils::secureLogValue(logger, \
      std::string(str) + " average, total (ms)", \
      std::to_string(glTimerQueries[name].averageMillis()) \
      + std::string(", ") \
      + std::to_string(glTimerQueries[name].totalMillis()) \
      );
#else
  #define TIMERS_DECLARE(...)
  #define TIMERS_CREATE()
  #define TIMERS_DESTROY()
  #define TIMERS_UPDATE()
  #define TIMER_TICK(name)
  #define TIMER_TOCK(name)
  #define TIMER_LOG(logger, name, str)
#endif // UTILS_QUERY_TIMERS
}