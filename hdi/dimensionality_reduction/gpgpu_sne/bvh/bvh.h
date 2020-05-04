#pragma once

#include <array>
#include "hdi/utils/abstract_log.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/3d_utils.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/cu_timer.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_layout.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_memory.h"

namespace hdi {
  namespace dr {  
    namespace bvh {
      class BVH {
      public:
        BVH();
        ~BVH();

        void init(const TsneParameters &params, GLuint posBuffer, unsigned nPos);
        void destr();
        void compute(const Bounds3D &bounds, bool logTimers = false);

        const BVHExtMemr& memr() const {
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
        unsigned _iter;

        TsneParameters _params;
        BVHLayout _layout;
        BVHIntMemr _intMemr;
        BVHExtMemr _extMemr;
        InteropResource _extPos;
        
        // Timer handles for kernel runtime performance measurements
        enum TimerType {
          TIMR_MORTON,      // Morton code generation timer 
          TIMR_SORT,        // Radix sort timer
          TIMR_LEAVES,      // Leaf node computation timer
          TIMR_NODES,       // Non-leaf node computation timer

          // Static enum length
          TimerTypeLength
        };
        utils::AbstractLog* _logger;
        std::array<CuTimer, TimerTypeLength> _timers;  // Timers for kernel perf.
      };
    }
  }
}