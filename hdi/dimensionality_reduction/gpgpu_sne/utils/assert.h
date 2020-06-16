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

#include <iostream>
#include <sstream>
#include <string>
#include <glad/glad.h>

namespace hdi {
  inline
  std::string glReadableError(GLenum err) {
    switch (err) {
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
    case GL_STACK_OVERFLOW:
      return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
      return "GL_STACK_UNDERFLOW";
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_CONTEXT_LOST:
      return "GL_CONTEXT_LOST";
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
    default:
      return "unknown";
    }
  }

  inline
  void assertGlDetail(const char *msg, const char *file, int line, bool abort = true) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
      std::stringstream ss;
      ss << "OpenGL error\n"
         << "  code     " << err << " (" << glReadableError(err) << ")\n"
         << "  message  " << msg << '\n'
         << "  file     " << file << '\n'
         << "  line     " << line;
      std::cerr << ss.str() << std::endl;
      if (abort) {
        std::exit(EXIT_FAILURE);
      }
    }
  }

  #define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
  #define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
  #define ASSERT_GL(msg) { assertGlDetail(msg, __FILE__, __LINE__); }

  inline
  void memoryGlDetail(const char *file, int line) {
    // This requires an NVX extension, but that's generally available I guess
    GLint total, free;
    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total);
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &free);

    // Adjust for megabytes
    total /= 1000;
    free /= 1000;

    float percFree = 100.f * (static_cast<float>(free) / static_cast<float>(total));
    std::stringstream ss;
        ss << "GL memory\n"
           << "  free   " << free << "mb / " << total << "mb (" << percFree << "%)\n"
           << "  loc    " << file << '\n'
           << "  line   " << line;
    std::cout << ss.str() << std::endl;
  }

  #define MEMORY_GL() { memoryGlDetail(__FILE__, __LINE__); }
} // namespace hdi