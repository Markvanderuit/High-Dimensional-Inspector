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

#include <algorithm>
#include <iostream>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include "hdi/data/panel_data.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/assert.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute_shaders.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/cuda/cub_interop.h"

using uint = unsigned;

constexpr uint kMax = 192;       // Would not recommend exceeeding this value for 1M+ vector datasets
constexpr uint nProbe = 12;      // Painfully large memory impact
constexpr uint nListMult = 4;   // Painfully large memory impact

namespace hdi {
  namespace dr {
    GpgpuHdCompute::GpgpuHdCompute()
    : _isInit(false),
      _logger(nullptr),
      _time_knn(0.f)
    { }

    GpgpuHdCompute::~GpgpuHdCompute()
    {
      if (_isInit) {
        destr();
      }
    }

    void GpgpuHdCompute::init(const TsneParameters &params)
    {
      _params = params;

      // Initialize shader program objects
      {
        for (auto &program : _programs) {
          program.create();
        }

        _programs(ProgramType::eSimilarity).addShader(COMPUTE, similarity_src);
        _programs(ProgramType::eExpandNeighbours).addShader(COMPUTE, expansion_src);
        _programs(ProgramType::eLayout).addShader(COMPUTE, layout_src);
        _programs(ProgramType::eNeighbours).addShader(COMPUTE, neighbours_src);

        for (auto &program : _programs) {
          program.build();
        }
      }

      // Initialize buffer object handles.
      // Allocation is performed in GpgpuHdCompute::compute() as the required
      // memory size is not yet known
      glCreateBuffers(_buffers.size(), _buffers.data());

      INIT_TIMERS();
      ASSERT_GL("GpgpuSneCompute::init()");
      _isInit = true;
    }

    void GpgpuHdCompute::destr()
    {
      for (auto &program : _programs) {
        program.destroy();
      }
      glDeleteBuffers(_buffers.size(), _buffers.data());
      DSTR_TIMERS();
      ASSERT_GL("GpgpuSneCompute::destr()");
      _isInit = false;
    }

    void GpgpuHdCompute::compute(const data::PanelData<float>& data)
    {
      // Data size, dimensionality, requested nearest neighbours
      const uint n = data.numDataPoints();
      const uint d = data.numDimensions();
      const uint k = std::min(kMax, 3 * static_cast<uint>(_params._perplexity) + 1);
      const float *points = data.getData().data();
      
      // Temporary memory for kNN computation
      // Stored host-side because (a) we move to OpenGL afterwards (b) interopability requires
      // double the memory, but FAISS will eat everything
      std::vector<float> knnSquareDistances(n * k);
      std::vector<faiss::Index::idx_t> knnIndices(n * k);
      
      // 1. 
      // Compute k-nearest-neighbours using FAISS library
      // Produce a fixed nr. (3 * perplexity + 1, usually) of nearest neighbours
      {
        utils::ScopedTimer<float, utils::Seconds> timer(_time_knn);
        utils::secureLog(_logger, "Performing kNN-search");
        
        // Nr. of inverted lists used by FAISS. O(sqrt(n)) is apparently reasonable
        // src: https://github.com/facebookresearch/faiss/issues/112
        uint nLists = nListMult * static_cast<uint>(std::sqrt(n)); 

        // Use a single GPU device. For now, just grab device 0 and pray
        faiss::gpu::StandardGpuResources faissResources;
        faiss::gpu::GpuIndexIVFFlatConfig faissConfig;
        faissConfig.device = 0;
        faissConfig.indicesOptions = faiss::gpu::INDICES_32_BIT;
        faissConfig.flatConfig.useFloat16 = true;
        
        // Construct search index
        // Inverted file flat list gives accurate results at significant memory overhead.
        faiss::gpu::GpuIndexIVFFlat faissIndex(
          &faissResources,
          d, 
          nLists,
          faiss::METRIC_L2, 
          faissConfig
        );
        faissIndex.setNumProbes(nProbe);
        faissIndex.train(n, points);
        faissIndex.add(n, points);

        // Perform actual search
        // Store results device cide in cuKnnSquaredDistances, cuKnnIndices, as the
        // rest of construction is performed on device as well.
        faissIndex.search(
          n,
          points,
          k,
          knnSquareDistances.data(),
          knnIndices.data()
        );
        
        // Tell FAISS to bugger off
        faissIndex.reset();
        faissIndex.reclaimMemory();

        utils::secureLog(_logger, "Performed kNN-search");
      }

      // Preset buffer data
      const std::vector<uint> empty(n * k, 0);
      const std::vector<uint> indices(knnIndices.begin(), knnIndices.end());

      // Create temporary buffer handles
      GLuint tempDistancesBuffer;
      GLuint tempNeighboursBuffer;
      GLuint tempSimilaritiesBuffer;
      GLuint tempSizesBuffer;
      GLuint tempScanBuffer;
      glCreateBuffers(1, &tempNeighboursBuffer);
      glCreateBuffers(1, &tempDistancesBuffer);
      glCreateBuffers(1, &tempSimilaritiesBuffer);
      glCreateBuffers(1, &tempSizesBuffer);
      glCreateBuffers(1, &tempScanBuffer);

      // Specify temporary buffers
      glNamedBufferStorage(tempDistancesBuffer, n * k * sizeof(float), knnSquareDistances.data(), 0);
      glNamedBufferStorage(tempNeighboursBuffer, n * k * sizeof(uint), indices.data(), 0);
      glNamedBufferStorage(tempSimilaritiesBuffer, n * k * sizeof(float), empty.data(), 0);
      glNamedBufferStorage(tempSizesBuffer, n * sizeof(uint), empty.data(), 0);
      glNamedBufferStorage(tempScanBuffer, n * sizeof(uint), nullptr, 0);
      
      // 2.
      // Compute similarities over the provided kNN data. This step is kind of prone to error
      // but is pretty much a direct copy of the original formulation used in BH-SNE, which is
      // also used exactly like this in Linear-complexity tSNE and CUDA-tSNE
      {
        TICK_TIMER(TIMR_SIMILARITY);
        
        // Set program uniforms
        auto &program = _programs(ProgramType::eSimilarity);
        program.bind();
        program.uniform1ui("nPoints", n);
        program.uniform1ui("kNeighbours", k);
        program.uniform1f("perplexity", _params._perplexity);
        program.uniform1ui("nIters", 200);
        program.uniform1f("epsilon", 1e-5);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempNeighboursBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempDistancesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempSimilaritiesBuffer);

        // Dispatch shader
        glDispatchCompute(ceilDiv(n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        ASSERT_GL("GpgpuSneCompute::compute::similarities()");
        TOCK_TIMER(TIMR_SIMILARITY);
      }

      // 3.
      // Expand the kNN so it is symmetric. That is, every neighbour referred by a point, itself
      // also refers to the point as its neighbour.
      {
        TICK_TIMER(TIMR_EXPANSION);
        
        auto &program = _programs(ProgramType::eExpandNeighbours);
        program.bind();
        program.uniform1ui("nPoints", n);
        program.uniform1ui("kNeighbours", k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempNeighboursBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempSizesBuffer);

        // Dispatch shader
        glDispatchCompute(ceilDiv(n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        ASSERT_GL("GpgpuSneCompute::compute::expansion()");
        TOCK_TIMER(TIMR_EXPANSION);
      }

      // 4.
      // Determine the sizes of the expanded neighbourshoods as well as the total required
      // size in memory though a prefix sum
      uint symmetricSize;
      {
        InteropInclusiveScanner scanner;
        scanner.init(tempSizesBuffer, tempScanBuffer, n);
        scanner.inclusiveScan(n);
        scanner.destr();

        glGetNamedBufferSubData(tempScanBuffer, (n - 1) * sizeof(uint), sizeof(uint), &symmetricSize);
      }

      // Specify permanent buffers
      glNamedBufferStorage(_buffers(BufferType::eLayout), n * 2 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighbours), symmetricSize * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float), nullptr, 0);
      glClearNamedBufferData(_buffers(BufferType::eLayout), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eNeighbours), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eSimilarities), GL_R32F, GL_RED, GL_FLOAT, nullptr);

      // 5.
      // Generate BufferType::eLayout contents
      {
        // Set program uniforms
        auto &program = _programs(ProgramType::eLayout);
        program.bind();
        program.uniform1ui("nPoints", n);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempScanBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

        // Dispatch shader
        glDispatchCompute(ceilDiv(n, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        ASSERT_GL("GpgpuSneCompute::compute::layout()");
      }

      // 6.
      // Generate the expanded BufferType::eNeighbours and BufferType::eSimilarities buffers
      // Compute symmetric neighbourhoods and similarities
      {
        TICK_TIMER(TIMR_SYMMETRIZE);

        // Clear sizes buffer, we recycle it as an atomic counter
        glClearNamedBufferData(tempSizesBuffer, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

        // Set program uniforms
        auto &program = _programs(ProgramType::eNeighbours);
        program.bind();
        program.uniform1ui("nPoints", n);
        program.uniform1ui("kNeighbours", k);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempNeighboursBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempSimilaritiesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempSizesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbours));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

        // Dispatch shader
        glDispatchCompute(ceilDiv(n * k, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        ASSERT_GL("GpgpuSneCompute::compute::neighbours()");
        TOCK_TIMER(TIMR_SYMMETRIZE);
      }

      // Delete temporary buffers
      glDeleteBuffers(1, &tempScanBuffer);
      glDeleteBuffers(1, &tempSizesBuffer);
      glDeleteBuffers(1, &tempSimilaritiesBuffer);
      glDeleteBuffers(1, &tempDistancesBuffer);
      glDeleteBuffers(1, &tempNeighboursBuffer);

      ASSERT_GL("GpgpuSneCompute::compute()");

      // Poll twice so timers are swapped
      POLL_TIMERS();
      POLL_TIMERS();

      // Report runtimes 
      utils::secureLog(_logger, "\nSimilarities Computation");
      utils::secureLogValue(_logger, "  kNN (ms)", 
        std::to_string(_time_knn * 1000));
      utils::secureLogValue(_logger, "  Expansion (ms)", 
        std::to_string(glTimers[TIMR_EXPANSION].lastMillis()));
      utils::secureLogValue(_logger, "  Similarity (ms)", 
        std::to_string(glTimers[TIMR_SIMILARITY].lastMillis()));
      utils::secureLogValue(_logger, "  Symmetrize (ms)", 
        std::to_string(glTimers[TIMR_SYMMETRIZE].lastMillis()));
      utils::secureLog(_logger, "");
    }
  } // namespace dr
} // namespace hdi