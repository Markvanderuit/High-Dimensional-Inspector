/*
 *
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
 *
 */

#ifndef SCOPED_TIMERS_H
#define SCOPED_TIMERS_H

#include <chrono>
#include <ctime>
#include "hdi/utils/timers.h"
#include "hdi/utils/timing_utils.h"

namespace hdi{
  namespace utils{

    template <class T, class UnitOfMeas = Milliseconds>
    class ScopedTimer {
    public:
      //! start the timer
      ScopedTimer(T& elapsed_time);
      //! stop the timer and save the elapsedTime
      ~ScopedTimer();

    private:
      Timer   _timer;
      T&     _elapsed_time;
    };

    template <class T, class UnitOfMeas = Milliseconds>
    class ScopedIncrementalTimer {
    public:
      //! start the timer
      ScopedIncrementalTimer(T& elapsed_time);
      //! stop the timer and save the elapsedTime
      ~ScopedIncrementalTimer();

    private:
      Timer   _timer;
      T&     _elapsed_time;
    };

    template <class T, class I = size_t, class UnitOfMeas = Milliseconds>
    class ScopedAveragingTimer {
    public:
      //! start the timer
      ScopedAveragingTimer(T& elapsed_time, I& iterations);
      //! stop the timer and save the elapsedTime
      ~ScopedAveragingTimer();

    private:
      Timer   _timer;
      T&     _elapsed_time;
      I&     _iterations;
    };
  }
}

//Implementation
#include "scoped_timers_inl.h"

#endif // SCOPED_TIMERS_H
