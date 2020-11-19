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
 * Log ouput parameters.
 * 
 * - avg timers: output average runtime of shaders throughout the program after minimization
 * - large allocs: output info when a large nr. of buffers is allocated (eg at BVH construction)
 * - field iter: output info on field computation each iteration
 * - field resize: output info on field resizing when this occurs
 */
#define LOG_AVG_TIMERS
#define LOG_LARGE_ALLOCS 
#define LOG_FIELD_ITER
#define LOG_FIELD_RESIZE

/**
 * Gradient descent parameters. We use two collections of values. The former (from HDI) provides
 * better results for small datasets such as MNIST/Fashion-MNIST, while the latter (from CUDA-tSNE) 
 * does the same for much larger datasets such as ImageNet and Word2Vec.
 */
// #define USE_CUDA_GRAD_PARAMS
#ifndef USE_CUDA_GRAD_PARAMS
#define GRAD_MOMENTUM 0.2f
#define GRAD_FINAL_MOMENTUM 0.5f
#define GRAD_EXAGGERATION_FACTOR 4.f
#else
#define GRAD_MOMENTUM 0.5f
#define GRAD_FINAL_MOMENTUM 0.8f
#define GRAD_EXAGGERATION_FACTOR 12.f
#endif

/**
 * Field resolution parameters. The size of the field and its growth over time heavily
 * influence the accuracy and runtime of the method. 2.0 is a good ratio for 2D, but
 * far too large for 3D as the field just becomes larger than N quickly. Keeping size 
 * fixed may be nice for debugging, but should never be used in a real application!
 * 
 * There are different parameters for 2D and 3D.
 */
#define FIELD_DO_ADAPTIVE_RESOLUTION
#define FIELD_FIXED_SIZE 40
#define FIELD_MIN_SIZE 5

/**
 * Dual hierarchy traversal parameters
 * 
 * - lvl difference: DH-traversal kicks in when both hierarchies are within x lvls of each other
 * - large leaf: large leaf pairs are pushed on a work queue for computation in separate shader
 */
#define DUAL_BVH_LVL_DIFFERENCE 3
#define DUAL_BVH_LARGE_LEAF 16

/**
 * Single hierarchy traversal parameters.
 * The wider subgroup-based traversal might be slightly faster, but its honestly too close.
 * 
 * There are different parameters for 2D and 3D.
 */
// #define EMB_BVH_WIDE_TRAVERSAL_2D
// #define EMB_BVH_WIDE_TRAVERSAL_3D

/**
 * Hierarchy construction parameters.
 * 
 * There are different parameters for 2D and 3D.
 * 
 * - kNode: node-level fan-out
 * - logk: log2(kNode), a value that is used almost everywhere
 * - kLeaf: leaf-level fan-out for the embedding hierarchy. Not used for field hierarchy.
 */
#ifndef EMB_BVH_WIDE_TRAVERSAL_2D
  #define BVH_KNODE_2D 4
  #define BVH_LOGK_2D 2
#else
  #define BVH_KNODE_2D 16
  #define BVH_LOGK_2D 4
#endif
#ifndef EMB_BVH_WIDE_TRAVERSAL_3D
  #define BVH_KNODE_3D 8
  #define BVH_LOGK_3D 3
#else
  #define BVH_KNODE_3D 16
  #define BVH_LOGK_3D 4
#endif
#define EMB_BVH_KLEAF_2D 4
#define EMB_BVH_KLEAF_3D 4