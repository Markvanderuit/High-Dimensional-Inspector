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

#include <cstdint>
#include <iostream>
#include <fstream>
#include <bitset>
#include <limits>
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "glsl_tester.h"

#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

GLSL(test_src, 450,
  const float nan = intBitsToFloat(0x7fffffff);

  layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  // layout(binding = 0, rgba32ui) coherent uniform uimage2D testImage;
  layout(binding = 0, std430) coherent buffer LockBuffer { uint lockValue; };
  layout(binding = 1, std430) coherent buffer OutBuffer { uint outValue; };

  void main() {
    // Simple spinlock
    // uint lock = 0u;
    // do {
    //   memoryBarrier();
    //   barrier();

    //   // Acquire lock
    //   lock = atomicCompSwap(lockValue, 0u, 1u);
    //   if (lock == 0u) {
    //     // Perform critical section
    //     outValue++;
    //     memoryBarrier();
    //     barrier();

    //     // Release lock
    //     atomicExchange(lockValue, 0u);
    //   }
    // } while (lock != 0u);
    // imageAtomicAdd(testImage, ivec2(0), 1u);
    outValue = uint(isnan(nan));
  }
);

namespace hdi::dr {
  GlslTester::GlslTester() 
  : _initialized(false), _logger(nullptr) { }

  GlslTester::~GlslTester() {
    if (_initialized) {
      clean();
    }
  }

  void GlslTester::initialize() {
    // Generate programs
    try {
      for (auto& program : _programs) {
        program.create();
      }
      _programs[PROGRAM_TEST].addShader(COMPUTE, test_src);
      for (auto& program : _programs) {
        program.build();
      }
    } catch (const ShaderLoadingException& e) {
      std::cerr << e.what() << std::endl;
      exit(0); // yeah no, not recovering from this catastrophy
    }

    // Generate buffers
    glGenBuffers(_buffers.size(), _buffers.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_LOCK]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 1 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_OUT]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 1 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

    // Generate textures
    glGenTextures(_textures.size(), _textures.data());

    GL_ASSERT("GlslTester::initialize");
    _initialized = true;
  }

  void GlslTester::compute() {
    // Run shader program
    {
      auto& program = _programs[PROGRAM_TEST];
      program.bind();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers[BUFFER_LOCK]);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers[BUFFER_OUT]);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.release();
      GL_ASSERT("GlslTester::compute::PROGRAM_TEST");
    }

    // Print shader output
    {
      std::vector<uint32_t> buffer(1, 0u);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, _buffers[BUFFER_OUT]);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 1 * sizeof(uint32_t), buffer.data());
      for (const auto& i : buffer) {
        std::cout << std::bitset<32>(i) << ", " << i << '\n';
      }
    }
  }

  void GlslTester::clean() {
    glDeleteBuffers(_buffers.size(), _buffers.data());
    glDeleteTextures(_textures.size(), _textures.data());
    for (auto& program : _programs) {
      program.destroy();
    }
    _initialized = false;
  }
}