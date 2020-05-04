
#include <bitset>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/helper_math.h>
#include <cub/cub.cuh>
#include <vector>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"
#include <cuda_gl_interop.h> // include last because of OpenGL header

/**
 * Integer division of n by div, but rounds up instead of down.
 * 
 * Only valid for positive numbers n > 0 and div > 0. Suspect to overflow
 * for very large n and div.
 * Source: https://stackoverflow.com/questions/2745074/fast-\ceiling-of-an-integer-division-in-c-c
 */
template <typename genType> 
inline
genType ceilDiv(genType n, genType div) {
  return (n + div - 1) / div;
}

#define gpuAssert(ans) { gpuAssertDetail((ans), __FILE__, __LINE__); }
inline void gpuAssertDetail(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"gpuAssert\t%s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
        exit(code);
      }
   }
}

struct SplitPointOp {
  template <typename T>
  __device__ __forceinline__
  T operator()(const T &a, const T &b) const {
      return (b < a) ? b : a;
  }
};

namespace hdi {
  namespace dr {  
    namespace bvh {
      BVH::BVH()
      : _isInit(false) { }

      BVH::~BVH() {
        if (_isInit) {
          destr();
        }
      }

      void BVH::init(const TsneParameters &params, GLuint posBuffer, unsigned nPos) {
        _params = params;
        _layout = BVHLayout(4, 8, nPos);

        // Fire up cub::RadixSort to figure out temporary memory size in bytes
        size_t tempSize;
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          nullptr, tempSize, nullptr, nullptr, nullptr, nullptr, _layout.nPos
        );
        
        // Initialize managing memory subclasses
        _intMemr.init(_layout, tempSize);
        _extMemr.init(_layout);
        _extPos.init(posBuffer);

        for (auto &timer : _timers) {
          timer.init();
        }
        _isInit = true;
      }

      void BVH::destr() {
        _extPos.destr();
        _extMemr.destr();
        _intMemr.destr();
        for (auto &timer : _timers) {
          timer.destr();
        }
        _isInit = false;
      }

      void BVH::compute(const Bounds3D &bounds, bool logTimers) {
        // Map external resources for access
        _extMemr.map();
        _extPos.map();
        
        // BVH bounds in float4 format
        float4 minb = make_float4(bounds.min.x, bounds.min.y, bounds.min.z, 0.f);
        float4 invr = make_float4(1.f) 
          / make_float4(bounds.range().x, bounds.range().y, bounds.range().z, 1.f);
        
        // Generate morton codes over unsorted embedding positions
        _timers[TIMR_MORTON].tick();
        kernConstrMorton<<<ceilDiv(_layout.nPos, 256u), 256u>>>(
          _layout, minb, invr,
          (float4 *) _extPos.ptr(),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn)
        );
        gpuAssert(cudaDeviceSynchronize());
        _timers[TIMR_MORTON].tock();

        // Perform radix sort on embedding positions using generated morton codes
        _timers[TIMR_SORT].tick();
        size_t tempSize = _intMemr.memrSize(BVHIntMemr::MemrType::eTemp);
        cub::DeviceRadixSort::SortPairs<uint, float4>(
          (void *) _intMemr.ptr(BVHIntMemr::MemrType::eTemp), 
          tempSize, 
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
          (float4 *)  _extPos.ptr(), 
          (float4 *)  _extMemr.ptr(BVHExtMemr::MemrType::ePos),
          (int) _layout.nPos, 
          0, 30
        );
        gpuAssert(cudaDeviceSynchronize());
        _timers[TIMR_SORT].tock();

        // Try to find a split point
        /* if (logTimers) {
          std::vector<uint> codebuffer(_layout.nPos);
          std::vector<float4> posBuffer(_layout.nPos);
          cudaMemcpy(codebuffer.data(), _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut), _layout.nPos * sizeof(uint), cudaMemcpyDeviceToHost);
          cudaMemcpy(posBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::ePos), _layout.nPos * sizeof(float4), cudaMemcpyDeviceToHost);

          uint first = codebuffer[0];
          uint last = codebuffer[codebuffer.size() - 1];
          uint diff = last ^ first;

          for (int i = 1; i < _layout.nPos; i++) {
            uint prev = codebuffer[i - 1];
            uint curr = codebuffer[i];


            std::cout << std::bitset<32>(codebuffer[i]) << "\t:\t"
                      << posBuffer[i].x << ", " << posBuffer[i].y << ", " << posBuffer[i].z
                      << '\n';
          }
        } */

        // Compute leaves
        _timers[TIMR_LEAVES].tick();
        kernConstrLeaves<<<ceilDiv(_layout.nLeaves, 256u), 256u>>>(
          _layout, 
          (float4 *)  _extMemr.ptr(BVHExtMemr::MemrType::ePos), // _pExtr[EXTR_POS], 
          (float4 *)  _extMemr.ptr(BVHExtMemr::MemrType::eNode), // _pExtr[EXTR_NODE],
          (uint *)    _extMemr.ptr(BVHExtMemr::MemrType::eMass), // _pExtr[EXTR_MASS], 
          (uint *)    _extMemr.ptr(BVHExtMemr::MemrType::eIdx), // _pExtr[EXTR_IDX], 
          (float4 *)  _extMemr.ptr(BVHExtMemr::MemrType::eMinB), //_pIntr[INTR_MIN_BND], 
          (float4 *)  _extMemr.ptr(BVHExtMemr::MemrType::eMaxB) //_pIntr[INTR_MAX_BND]
        );
        gpuAssert(cudaDeviceSynchronize());
        _timers[TIMR_LEAVES].tock();

        _timers[TIMR_NODES].tick();
        {
          const uint logk = static_cast<uint>(std::log2(_layout.kNode));
          uint offset = _layout.nNodes + _layout.nLeaves;
          for (uint level = _layout.nLvls - 1; level > 0; level--) {
            const uint nNodes = 1u << (logk * level);
            offset -= nNodes;
            if (_layout.kNode == 2u) {
              kernConstrNodes<2><<<ceilDiv(nNodes, 256u), 256>>>(
                _layout, level, nNodes, offset,
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
                (uint *)   _extMemr.ptr(BVHExtMemr::MemrType::eMass),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMaxB)
              );
            } else if (_layout.kNode == 4u) {
              kernConstrNodes<4><<<ceilDiv(nNodes, 256u), 256>>>(
                _layout, level, nNodes, offset,
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
                (uint *)   _extMemr.ptr(BVHExtMemr::MemrType::eMass),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMaxB)
              );
            } else if (_layout.kNode == 8u) {
              kernConstrNodes<8><<<ceilDiv(nNodes, 256u), 256>>>(
                _layout, level, nNodes, offset,
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
                (uint *)   _extMemr.ptr(BVHExtMemr::MemrType::eMass),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMaxB)
              );
            } else if (_layout.kNode == 16u) {
              kernConstrNodes<16><<<ceilDiv(nNodes, 256u), 256>>>(
                _layout, level, nNodes, offset,
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
                (uint *)   _extMemr.ptr(BVHExtMemr::MemrType::eMass),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
                (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMaxB)
              );
            }
          }
        }
        gpuAssert(cudaDeviceSynchronize());
        _timers[TIMR_NODES].tock();

        // Perform subdivision
        // ...
        // gpuAssert(cudaDeviceSynchronize());
        
        // Construct tree bounds, diams
        // ...
        // gpuAssert(cudaDeviceSynchronize());
        
        // Unmap external resources for access
        _extMemr.unmap();
        _extPos.unmap();
        
        for (auto &timer : _timers) {
          timer.poll();
        }

        if (logTimers) {
          utils::secureLog(_logger, "BVH construction");
          _timers[TIMR_MORTON].log(_logger, "  Morton");
          _timers[TIMR_SORT].log(_logger, "  Sorting");
          _timers[TIMR_LEAVES].log(_logger, "  Leaves");
          _timers[TIMR_NODES].log(_logger, "  Nodes");
          utils::secureLog(_logger, "");
        }
      }
    }
  }
}