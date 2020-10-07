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

/**
 * GLSL(name, version, ...)
 * 
 * GLSL verbatism wrapper, embedding GLSL as a string literal in C++ code. If
 * preprocessor code is involved in GLSL, this must be wrapped in a GLSL_PROTECT()
 * statement.
 *  
 * @author M. Billeter
 * 
 * @param name - name of string that becomes embedded
 * @param version - required version of shader, eg. 450
 * @param ... - misc arguments contain shader code.
 */
#define GLSL(name, version, ...) \
  static const char * name = \
  "#version " #version "\n" GLSL_IMPL(__VA_ARGS__)
#define GLSL_IMPL(...) #__VA_ARGS__

/**
 * GLSL_PROTECT(...)
 * 
 * Helper wrapper to allow embedding of GLSL preprocessor statements
 * (eg. #extension) in embedded GLSL code.
 * 
 * @author M. Billeter
 * 
 * @param ... - statement to wrap
 */
#define GLSL_PROTECT(...) \n __VA_ARGS__ \n

