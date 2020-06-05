
#include <bitset>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/helper_math.h>
#include <cub/cub.cuh>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/bvh/bvh_kern.h"
#include <cuda_gl_interop.h> // include last because of OpenGL header

std::string to_string(float4 f)
{
  std::stringstream ss;
  ss << '{' << f.x << ',' << f.y << ',' << f.z << ',' << f.w << '}';
  return ss.str();
}

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
inline 
void gpuAssertDetail(cudaError_t code, const char *file, int line, bool abort=true) {
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
      : _isInit(false),
      _iter(0) { }

      BVH::~BVH() {
        if (_isInit) {
          destr();
        }
      }

      void BVH::init(const TsneParameters &params, GLuint posBuffer, unsigned nPos) {
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
          float4 rootNode = make_float4(0.f, 0.f, 0.f, _layout.nPos);
          cudaMemcpy(_extMemr.ptr(BVHExtMemr::MemrType::eIdx),
            &rootIdx, sizeof(uint), cudaMemcpyHostToDevice);
          cudaMemcpy(_extMemr.ptr(BVHExtMemr::MemrType::eNode),
            &rootNode, sizeof(float4), cudaMemcpyHostToDevice);
        }
        _extMemr.unmap();

        for (auto &timer : _timers) {
          timer.init();
        }
        _isInit = true;
        _iter = 0;
      }

      void BVH::destr() {
        _extPos.destr();
        _extMemr.destr();
        _intMemr.destr();
        for (auto &timer : _timers) {
          timer.destr();
        }
        _isInit = false;
        _iter = 0;
      }

      void BVH::compute(const Bounds3D &bounds, bool logTimers) {
        // Map external resources for access
        _extMemr.map();
        _extPos.map();
        
        // Generate 30-bit morton codes over unsorted embedding positions
        _timers[TIMR_MORTON].tick();
        // BVH bounds in float4 format
        float4 minb = make_float4(bounds.min.x, bounds.min.y, bounds.min.z, 0.f);
        float4 invr = make_float4(1.f) 
          / make_float4(bounds.range().x, bounds.range().y, bounds.range().z, 1.f);
        kernConstrMorton<<<ceilDiv(_layout.nPos, 256u), 256u>>>(
          _layout, minb, invr,
          (float4 *) _extPos.ptr(),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn)
        );
        _timers[TIMR_MORTON].tock();

        // Perform radix sort on embedding positions using 30-bit morton codes
        _timers[TIMR_SORT].tick();
        size_t tempSize = _intMemr.memrSize(BVHIntMemr::MemrType::eTemp);
        cub::DeviceRadixSort::SortPairs<uint, uint>(
          (void *) _intMemr.ptr(BVHIntMemr::MemrType::eTemp), 
          tempSize, 
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonIn),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxIn),
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxOut),
          (int) _layout.nPos, 
          0, 30
        );
        kernConstrPos<<<ceilDiv(_layout.nPos, 256u), 256u>>>(
          _layout,
          (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eIdxOut),
          (float4 *) _extPos.ptr(), 
          (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::ePos)
        );
        _timers[TIMR_SORT].tock();

        _timers[TIMR_SUBDIV].tick();
        // if (_iter <= _params._remove_exaggeration_iter || _iter % 4 == 0) {
          // Perform subdivision and construct leaf nodes
          for (uint l = 1; l < _layout.nLvls; l++) {
            const uint begin = 1 << l;
            const uint end = 2 * begin;
            kernConstrSubdiv<<<ceilDiv(end - begin, 256u), 256u>>>(
              _layout,
              l, _layout.nLvls, begin, end,
              (uint *) _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
              (uint *) _extMemr.ptr(BVHExtMemr::MemrType::eIdx),
              (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::ePos),
              (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
              (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
              (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eDiam)
            );
          }
        // }
        _timers[TIMR_SUBDIV].tock();
        
        // Compute additional tree data bottom-up
        _timers[TIMR_DATA].tick();
        for (int l = _layout.nLvls - 1; l > 0; l -= 6) {
          const uint begin = 1 << l;
          const uint end = 2 * begin;
          kernConstrDataReduce<<<ceilDiv(end - begin, 64u), 64u>>>(
            _layout, 
            l, _layout.nLvls, begin, end,
            (uint *) _extMemr.ptr(BVHExtMemr::MemrType::eIdx),
            (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::ePos),
            (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
            (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
            (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eDiam)
          );
        }
        _timers[TIMR_DATA].tock();

        _timers[TIMR_CLEANUP].tick();
        kernConstrCleanup<<<ceilDiv(_layout.nNodes, 1024u), 1024u>>>(
          _layout.nNodes,
          (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eNode),
          (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
          (float4 *) _extMemr.ptr(BVHExtMemr::MemrType::eDiam)
        );
        _timers[TIMR_CLEANUP].tock();

        /* {
          cudaDeviceSynchronize();

          std::vector<uint> mortonBuffer(_layout.nPos);
          std::vector<uint> idxBuffer(_layout.nNodes);
          std::vector<uint> massBuffer(_layout.nNodes);
          std::vector<float4> nodeBuffer(_layout.nNodes);
          std::vector<float4> minbBuffer(_layout.nNodes);
          std::vector<float4> maxbBuffer(_layout.nNodes);

          cudaMemcpy(mortonBuffer.data(), _intMemr.ptr(BVHIntMemr::MemrType::eMortonOut),
            mortonBuffer.size() * sizeof(uint), cudaMemcpyDeviceToHost);
          cudaMemcpy(idxBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::eIdx),
            idxBuffer.size() * sizeof(uint), cudaMemcpyDeviceToHost);
          cudaMemcpy(massBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::eMass),
            massBuffer.size() * sizeof(uint), cudaMemcpyDeviceToHost); 
          cudaMemcpy(nodeBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::eNode),
            nodeBuffer.size() * sizeof(float4), cudaMemcpyDeviceToHost); 
          cudaMemcpy(minbBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::eMinB),
            minbBuffer.size() * sizeof(float4), cudaMemcpyDeviceToHost); 
          cudaMemcpy(maxbBuffer.data(), _extMemr.ptr(BVHExtMemr::MemrType::eMaxB),
            maxbBuffer.size() * sizeof(float4), cudaMemcpyDeviceToHost); 

          for (uint i = 0; i < _layout.nPos; i++) {
            std::cout << i << " : " << std::bitset<32>(mortonBuffer[i]) << '\n';
          }

          for (uint i = 0; i < _layout.nNodes; i++) {
            std::cout << i << " - "
                      << "idx : " << idxBuffer[i] << "  -  "
                      << "mass : " << massBuffer[i] << "  -  "
                      << "node : " << to_string(nodeBuffer[i]) << "  -  "
                      << "min : " << to_string(minbBuffer[i]) << "  -  "
                      << "max : " << to_string(maxbBuffer[i]) << '\n';
          }
          
          exit(0);
        } */
        
        // Unmap external resources for access
        _extMemr.unmap();
        _extPos.unmap();
        
        for (auto &timer : _timers) {
          timer.poll();
        }

        _iter++;

        if (logTimers) {
          utils::secureLog(_logger, "BVH construction");
          _timers[TIMR_MORTON].log(_logger, "  Morton");
          _timers[TIMR_SORT].log(_logger, "  Sorting");
          _timers[TIMR_SUBDIV].log(_logger, "  Subdiv");
          _timers[TIMR_DATA].log(_logger, "  Bboxes");
          _timers[TIMR_CLEANUP].log(_logger, "  Cleanup");
          utils::secureLog(_logger, "");
        }
      }
    }
  }
}