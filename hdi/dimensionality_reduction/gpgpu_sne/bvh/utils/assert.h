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

#include <cstdlib>
#include <cuda_runtime.h>

namespace hdi {
  namespace dr {
    namespace bvh {
      inline 
      void assertCuDetail(cudaError_t err, const char *file, int line, bool abort=true) {
        if (err != cudaSuccess) {
          std::stringstream ss;
          ss << "CUDA error\n"
            << "  code     " << err << " (" << cudaGetErrorString(err) << ")\n"
            << "  file     " << file << '\n'
            << "  line     " << line;
          std::cerr << ss.str() << std::endl;
          if (abort) {
            std::exit(EXIT_FAILURE);
          }
        }
      }

      #define ASSERT_CU(err) { assertCuDetail(err, __FILE__, __LINE__); }

      inline
      void memoryCuDetail(const char *file, int line) {
        std::size_t free, total;
        cudaMemGetInfo(&free, &total);

        // Adjust for megabytes
        free /= 1000'000;
        total /= 1000'000;

        float percFree = 100.f * (static_cast<float>(free) / static_cast<float>(total));

        std::stringstream ss;
        ss << "CUDA memory\n"
           << "  free     " << free << "mb / " << total << "mb (" << percFree << "%)\n"
           << "  loc      " << file << '\n'
           << "  line     " << line;
        std::cout << ss.str() << std::endl;
      }
      
      #define MEMORY_CU() { memoryCuDetail(__FILE__, __LINE__); }
    }
  }
}