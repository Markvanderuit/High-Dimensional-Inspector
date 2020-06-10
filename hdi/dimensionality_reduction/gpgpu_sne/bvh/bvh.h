#pragma once

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_memory.h"

namespace hdi {
  namespace dr {  
    namespace bvh {
      template <unsigned D>
      class BVH {
      public:
        BVH();
        ~BVH();

        void init(const TsneParameters &params,
                  GLuint posBuffer,
                  GLuint boundsBuffer,
                  unsigned nPos);
        void destr();
        void compute(bool rebuild, unsigned iteration);

        const BVHExtMemr<D>& memr() const {
          return _extMemr;
        }

        const BVHLayout& layout() const {
          return _layout;
        }

        void setLogger(utils::AbstractLog* logger) {
          _logger = logger; 
        }

      private:
        bool _isInit;
 
        TsneParameters _params;
        BVHLayout _layout;
        BVHIntMemr _intMemr;
        BVHExtMemr<D> _extMemr;
        InteropResource _extPos;
        InteropResource _extBounds;
        
        // Timer handles for kernel runtime performance measurements
        enum TimerType {
          TIMR_MORTON,      // Morton code generation timer 
          TIMR_SORT,        // Radix sort timer
          TIMR_SUBDIV,      // Tree subdivision timer
          TIMR_DATA,        // Tree data computation timer
          TIMR_CLEANUP,

          // Static enum length
          TimerTypeLength
        };
        utils::AbstractLog* _logger;
        std::array<CuTimer, TimerTypeLength> _timers;  // Timers for kernel perf.
      };
    }
  }
}