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
#include <deque>
#include <numeric>
#include <stdexcept>
#include <glad/glad.h>

#define GL_TIMERS_ENABLED // Enable/disable GL timers globally

namespace hdi::dr {
#ifdef GL_TIMERS_ENABLED
  class GlTimer {
  public:
    GlTimer()
    : _isInit(false) { }

    ~GlTimer() {
      if (_isInit) {
        dstr();
      }
    }

    void init() {
      glCreateQueries(GL_TIME_ELAPSED, 1, &_handle);
      _values.fill(0);
      _iteration = 0;
      _isInit = true;
    }

    void dstr() {
      glDeleteQueries(1, &_handle);
      _values.fill(0);
      _iteration = 0;
      _isInit = false;
    }

    void tick() {
      glBeginQuery(GL_TIME_ELAPSED, _handle);
    }
    
    void tock() {
      glEndQuery(GL_TIME_ELAPSED);
    }

    bool poll() {
      int i = 0;
      glGetQueryObjectiv(_handle, GL_QUERY_RESULT_AVAILABLE, &i);
      if (!i) {
        return GL_FALSE;
      }
      
      glGetQueryObjecti64v(_handle, GL_QUERY_RESULT, &_values[TIMR_LAST_QUERY]);
      _values[TIMR_TOTAL] += _values[TIMR_LAST_QUERY];
      _values[TIMR_AVERAGE] = _values[TIMR_AVERAGE] 
                             + (_values[TIMR_LAST_QUERY] - _values[TIMR_AVERAGE]) 
                             / (++_iteration);

      return GL_TRUE;
    }

    double lastMillis() const {
      return static_cast<double>(_values[TIMR_LAST_QUERY]) / 1000000.0;
    }

    double lastMicros() const {
      return static_cast<double>(_values[TIMR_LAST_QUERY]) / 1000.0;
    }

    double averageMillis() const {
      return static_cast<double>(_values[TIMR_AVERAGE]) / 1000000.0;
    }

    double averageMicros() const {
      return static_cast<double>(_values[TIMR_AVERAGE]) / 1000.0;
    }

    double totalMillis() const {
      return static_cast<double>(_values[TIMR_TOTAL]) / 1000000.0;
    }

    double totalMicros() const {
      return static_cast<double>(_values[TIMR_TOTAL]) / 1000.0;
    }

  private:
    enum ValueType {
      TIMR_LAST_QUERY,
      TIMR_AVERAGE,
      TIMR_TOTAL,

      // Static enum length
      ValueTypeLength 
    };

    bool _isInit;
    GLuint _handle;
    unsigned _iteration;
    std::array<GLint64, ValueTypeLength> _values;
  };

  #define DECL_TIMERS(...) \
    enum GlTimerType { \
      __VA_ARGS__, \
      GlTimerTypeLength \
    }; \
    std::array<GlTimer, GlTimerTypeLength> glTimers;
  
  #define INIT_TIMERS() \
    for (auto &t : glTimers) t.init();

  #define DSTR_TIMERS() \
    for (auto &t : glTimers) t.dstr();
    
  #define TICK_TIMER(name) \
    glTimers[name].tick();

  #define TOCK_TIMER(name) \
    glTimers[name].tock();

  #define POLL_TIMERS() \
    { \
      std::deque<int> t(GlTimerTypeLength); \
      std::iota(t.begin(), t.end(), 0); \
      while (!t.empty()) { \
        const int _t = t.front(); \
        t.pop_front(); \
        if (!glTimers[_t].poll()) { \
          t.push_back(_t); \
        } \
      } \
    }

#ifdef _WIN32
  #define LOG_TIMER(logger, name, str) \
    utils::secureLogValue(logger, \
      std::string(str) + " average (\xE6s)", \
      std::to_string(glTimers[name].averageMicros()) \
      );
#else
  #define LOG_TIMER(logger, name, str) \
      utils::secureLogValue(logger, \
        std::string(str) + " average (\xC2\xB5s)", \
        std::to_string(glTimers[name].averageMicros()) \
        );
#endif
#else
  #define DECL_TIMERS(...)
  #define INIT_TIMERS()
  #define DSTR_TIMERS()
  #define POLL_TIMERS()
  #define TICK_TIMERS(name)
  #define TOCK_TIMERS(name)
  #define LOG_TIMER(logger, name, str)
#endif
}