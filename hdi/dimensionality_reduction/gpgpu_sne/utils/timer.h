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

#include <algorithm>
#include <numeric>
#include "hdi/dimensionality_reduction/gpgpu_sne/constants.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/enum.h"

namespace hdi::dr {
  class GLtimer {
  public:
    // Types of values the timer can output
    enum class ValueType {
      eLast,    // Last recorded time
      eAverage, // Average of recorded times
      eTotal,   // Sum of recorded times

      Length
    };

    // Time scales for values the timer can output
    enum class TimeScale {
      eSeconds,
      eMillis,
      eMicros
    };

  public:
    GLtimer()
    : _isInit(false) { }

    ~GLtimer() {
      if (_isInit) {
        dstr();
      }
    }

    void init() {
      glCreateQueries(GL_TIME_ELAPSED, _handles.size(), _handles.data());
      _values.fill(0);
      _iteration = 0;
      _isInit = true;
    }

    void dstr() {
      if (!_isInit) {
        return;
      }

      glDeleteQueries(_handles.size(), _handles.data());
      _values.fill(0);
      _iteration = 0;
      _isInit = false;
    }

    void tick() {
      glBeginQuery(GL_TIME_ELAPSED, _handles(HandleType::eFront));
    }
    
    void tock() {
      glEndQuery(GL_TIME_ELAPSED);
    }

    void poll() {
      glGetQueryObjecti64v(_handles(HandleType::eBack), GL_QUERY_RESULT, &_values(ValueType::eLast));

      // Incrementally update total and average stored values
      _values(ValueType::eTotal) += _values(ValueType::eLast);
      _values(ValueType::eAverage) = _values(ValueType::eAverage) 
                                   + (_values(ValueType::eLast) - _values(ValueType::eAverage)) 
                                   / (++_iteration);

      // Swap front and back timers
      std::swap(_handles(HandleType::eFront), _handles(HandleType::eBack));
    }

    template <ValueType type, TimeScale scale>
    double get() const {
      constexpr double div = (scale == TimeScale::eMicros) ? 1'000.0
                           : (scale == TimeScale::eMillis) ? 1'000'000.0
                           : 1'000'000'000.0;
      return static_cast<double>(_values(type)) / div;
    }

  private:
    enum class HandleType {
      eFront,
      eBack,

      Length
    };

    bool _isInit;
    unsigned _iteration;
    EnumArray<HandleType, GLuint> _handles;
    EnumArray<ValueType, GLint64> _values;
  };

  inline
  void glCreateTimers(GLsizei n, GLtimer *timers) {
    for (GLsizei i = 0; i < n; ++i) {
      timers[i].init();
    }
  }

  inline
  void glDeleteTimers(GLsizei n, GLtimer *timers) {
    for (GLsizei i = 0; i < n; ++i) {
      timers[i].dstr();
    }
  }

  inline
  void glPollTimers(GLsizei n, GLtimer *timers) {
    for (GLsizei i = 0; i < n; ++i) {
      timers[i].poll();
    }
  }

#ifdef _WIN32
  #define LOG_TIMER(logger, name, str) \
    utils::secureLogValue(logger, \
      std::string(str) + " average (\xE6s)", \
      std::to_string(_timers(name).get<GLtimer::ValueType::eAverage, GLtimer::TimeScale::eMicros>()) \
    );
#else
  #define LOG_TIMER(logger, name, str) \
      utils::secureLogValue(logger, \
        std::string(str) + " average (\xC2\xB5s)", \
        std::to_string(_timers(name).get<GLtimer::ValueType::eAverage, GLtimer::TimeScale::eMicros>()) \
      );
#endif
}