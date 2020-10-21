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

#include "hdi/utils/abstract_log.h"

namespace hdi {
  namespace dr {
    namespace bvh {
      struct CuTimer {
      private:
        bool _isInit;
        unsigned _iter;
        float _msPoll, _msAvg;
        void *_start, *_stop;

      public:
        CuTimer();

        // Manage object lifetime
        void init();
        void destr();

        // Tick to start recording a timed sequence
        void tick();

        // Tick to stop recording a timed sequence
        void tock();

        // Poll until timer results are available
        void poll();

        float lastMillis() const
        {
          return _msPoll;
        }
        
        float lastMicros() const
        {
          return _msPoll * 1000.f;
        }

        float averageMillis() const
        {
          return _msAvg;
        }

        float averageMicros() const
        {
          return _msAvg * 1000.f;
        }
      };

      #ifdef _WIN32
        #define CU_LOG_TIMER(logger, timer, str) \
          utils::secureLogValue(logger, \
            std::string(str) + " average (\xE6s)", \
            std::to_string(timer.averageMicros()) \
          );
      #else
        #define CU_LOG_TIMER(logger, timer, str) \
          utils::secureLogValue(logger, \
            std::string(str) + " average (\xC2\xB5s)", \
            std::to_string(timer.averageMicros()) \
          );
      #endif
    }
  }
}