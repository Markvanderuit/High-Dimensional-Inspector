
#include <bitset>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/helper_math.h>
#include <cub/cub.cuh>
#include <numeric>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"
#include <cuda_gl_interop.h> // include last because of OpenGL header

template <unsigned D>
using vec = std::conditional<D == 2, float2, float4>::type;

template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

inline 
void gpuAssertDetail(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error\n"
    << "  code     " << code << '\n'
    << "  message  " << cudaGetErrorString(code) << '\n'
    << "  file     " << file << '\n'
    << "  line     " << line;
    if (abort) {
      exit(code);
    }
  }
}

#define gpuAssert(ans) { gpuAssertDetail((ans), __FILE__, __LINE__); }

namespace hdi {
  namespace dr {  
    namespace bvh {
      template <unsigned D>
      BVH<D>::BVH()
      : _isInit(false)
      { }

      template <unsigned D>
      BVH<D>::~BVH() {
        if (_isInit) {
          destr();
        }
      }

      template <unsigned D>
      void BVH<D>::init(const TsneParameters &params, GLuint posBuffer, GLuint boundsBuffer, unsigned nPos) {
        _params = params;
        _layout = BVHLayout(2, nPos);

        // Fire up cub::RadixSort to figure out temporary memory size in bytes
        size_t tempSize;
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          nullptr, tempSize, nullptr, nullptr, nullptr, nullptr, _layout.nPos
        );
        
        // Initialize managing memory subclasses
        _intMemr.init(_layout, tempSize);
        _extMemr.init(_layout);
        _extPos.init(posBuffer);
        _extBounds.init(boundsBuffer);

        // Fill unsorted indices
        {
          std::vector<uint> idx(nPos);
          std::iota(idx.begin(), idx.end(), 0u);
          cudaMemcpy(_intMemr.ptr(BVHIntMemr::MemrType::eIdxIn),
            idx.data(), sizeof(uint) * idx.size(), cudaMemcpyHostToDevice);
        }
        
        // Set root node range, as this (REALLY) shouldn't change
        _extMemr.map();
        {
          uint rootIdx = 0;
          float4 rootNode = make_float4(0.f, 0.f, 0.f, static_cast<float>(_layout.nPos));
          cudaMemcpy(_extMemr.ptr(BVHExtMemr<D>::MemrType::eIdx),
            &rootIdx, sizeof(uint), cudaMemcpyHostToDevice);
          cudaMemcpy(_extMemr.ptr(BVHExtMemr<D>::MemrType::eNode),
            &rootNode, sizeof(float4), cudaMemcpyHostToDevice);
        }
        _extMemr.unmap();

        for (auto &timer : _timers) {
          timer.init();
        }
        _isInit = true;
      }

      template <unsigned D>
      void BVH<D>::destr() {
        _extPos.destr();
        _extBounds.destr();
        _extMemr.destr();
        _intMemr.destr();
        for (auto &timer : _timers) {
          timer.destr();
        }
        _isInit = false;
      }

      template <unsigned D>
      void BVH<D>::compute(bool rebuild, unsigned iteration) {
        using vec = vec<D>;

        // Map external resources for access
        _extMemr.map();
        _extPos.map();
        _extBounds.map();

        // bool rebuild = iteration <= _params._remove_exaggeration_iter || iteration % 8 == 0;
        
        // Generate 30-bit morton codes over unsorted embedding positions
        _timers[TIMR_MORTON].tick();
        if (rebuild) {
          kernConstrMorton<D><<<ceilDiv(_layout.nPos, 256u), 256u>>>(
            _layout,
            (BVHBounds<D> *) _extBounds.ptr(),
            (vec *) _extPos.ptr(),
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn)
          );
        }
        _timers[TIMR_MORTON].tock();

        // Perform radix sort on embedding positions using 30-bit morton codes
        _timers[TIMR_SORT].tick();
        if (rebuild) {
          size_t tempSize = _intMemr.memrSize(BVHIntMemr::MemrType::eTemp);
          const int msb = 30;
          const int lsb = msb - _layout.nLvls;
          cub::DeviceRadixSort::SortPairs<uint, uint>(
            (void *) _intMemr.ptr(BVHIntMemr::MemrType::eTemp), 
            tempSize, 
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn),
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxIn),
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxOut),
            (int) _layout.nPos, 
            lsb, msb
          );
          kernConstrPos<D><<<ceilDiv(_layout.nPos, 256u), 256u>>>(
            _layout,
            (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxOut),
            (vec *) _extPos.ptr(), 
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::ePos)
          );
        }
        _timers[TIMR_SORT].tock();

        _timers[TIMR_SUBDIV].tick();
        if (rebuild) {
          // Perform subdivision and construct leaf nodes
          for (uint l = 1; l < _layout.nLvls; l++) {
            const uint begin = 1 << l;
            const uint end = 2 * begin;
            kernConstrSubdiv<D><<<ceilDiv(end - begin, 256u), 256u>>>(
              _layout,
              l, _layout.nLvls, begin, end,
              (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
              (uint *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eIdx),
              (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::ePos),
              (float4 *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eNode),
              (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eMinB),
              (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eDiam)
            );
          }
        }
        _timers[TIMR_SUBDIV].tock();
        
        // Compute additional tree data bottom-up
        _timers[TIMR_DATA].tick();
        {
          const uint begin = 1 << (_layout.nLvls - 1);
          const uint end = 2 * begin;
          kernConstrLeaf<D><<<ceilDiv(_layout.nNodes, 256u), 256u>>>(
            _layout,
            begin, end,
            (uint *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eIdx),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::ePos),
            (float4 *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eNode),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eMinB),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eDiam)
          );
        }

        for (int l = _layout.nLvls - 1; l > 0; l -= 6) {
          const uint begin = 1 << l;
          const uint end = 2 * begin;
          kernConstrBbox<D><<<ceilDiv(end - begin, 64u), 64u>>>(
            _layout, 
            l, _layout.nLvls, begin, end,
            (uint *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eIdx),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::ePos),
            (float4 *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eNode),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eMinB),
            (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eDiam)
          );
        }
        _timers[TIMR_DATA].tock();

        _timers[TIMR_CLEANUP].tick();
        /* kernConstrCleanup<D><<<ceilDiv(_layout.nNodes, 1024u), 1024u>>>(
          _layout.nNodes,
          (float4 *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eNode),
          (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eMinB),
          (vec *) _extMemr.ptr(BVHExtMemr<D>::MemrType::eDiam)
        ); */
        _timers[TIMR_CLEANUP].tock();
        
        // Unmap external resources for access
        _extMemr.unmap();
        _extBounds.unmap();
        _extPos.unmap();
        
        for (auto &timer : _timers) {
          timer.poll();
        }

        if (iteration >= static_cast<unsigned>(_params._iterations) - 1) {
          utils::secureLog(_logger, "\nBVH construction");
          _timers[TIMR_MORTON].log(_logger, "  Morton");
          _timers[TIMR_SORT].log(_logger, "  Sorting");
          _timers[TIMR_SUBDIV].log(_logger, "  Subdiv");
          _timers[TIMR_DATA].log(_logger, "  Bboxes");
          _timers[TIMR_CLEANUP].log(_logger, "  Cleanup");
        }
      }
      
      // Explicit template instantiations
      template class BVH<2>;
      template class BVH<3>;
    }
  }
}