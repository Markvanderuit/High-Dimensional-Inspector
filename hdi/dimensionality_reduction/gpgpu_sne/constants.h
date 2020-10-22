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

// Use wider subgroup-based traversal for embedding hierarchy in 2/3d
// A bit finicky and unpredictable
// #define EMB_BVH_2D_WIDE_TRAVERSAL
// #define EMB_BVH_3D_WIDE_TRAVERSAL

// Node fan-out for embedding/field hierarchy in 2d
// as well as log2 of said value
#ifndef EMB_BVH_2D_WIDE_TRAVERSAL
  #define BVH_2D_KNODE 4
  #define BVH_2D_LOGK 2
#else
  #define BVH_2D_KNODE 16
  #define BVH_2D_LOGK 4
#endif

#ifndef EMB_BVH_3D_WIDE_TRAVERSAL
  // Node fan-out for embedding/field hierarchy in 3d
  // as well as log2 of said value
  #define BVH_3D_KNODE 8
  #define BVH_3D_LOGK 3
#else
  // Node fan-out for embedding/field hierarchy in 3d
  // as well as log2 of said value
  #define BVH_3D_KNODE 8
  #define BVH_3D_LOGK 3
#endif

// Leaf fan-out for embedding hierarchy in 2/3d
#define EMB_BVH_2D_KLEAF 4
#define EMB_BVH_3D_KLEAF 4

// Any leaf with mass > 16 is not interacted with during traversal
// and is instead computed in a separate shader
#define EMB_BVH_LARGE_LEAF 16