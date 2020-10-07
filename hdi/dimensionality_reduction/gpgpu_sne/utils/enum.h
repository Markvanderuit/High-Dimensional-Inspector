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

#include <array>
#include <type_traits>
#include <glad/glad.h>

namespace hdi {
  namespace dr {
    // Cast enum value to underlying type of enum class (eg. int)
    template <typename ETy>
    constexpr inline
    typename std::underlying_type<ETy>::type underlying(ETy e) noexcept {
      return static_cast<typename std::underlying_type<ETy>::type>(e);
    }
    
    // Array class using Enums classes as indices
    // For a used enum E, E::Length must be specified
    template <typename ETy, typename Ty>
    class EnumArray : public std::array<Ty, underlying(ETy::Length)> {
    public:
      constexpr inline
      const Ty& operator()(ETy e) const {
        return this->operator[](underlying<ETy>(e));
      }

      constexpr inline
      Ty& operator()(ETy e) {
        return this->operator[](underlying<ETy>(e));
      }
    };

    // For an EnumArray of GLuints representing OpenGL buffer objects,
    // report the total memory range in machine units occupied by those buffer objects.
    template <typename ETy>
    GLuint bufferSize(const EnumArray<ETy, GLuint> &array) {
      GLuint size = 0;
      for (const auto &handle : array) {
        GLint i = 0;
        glGetNamedBufferParameteriv(handle, GL_BUFFER_SIZE, &i);
        size += static_cast<GLuint>(i);
      }
      return size;
    }
  }
}