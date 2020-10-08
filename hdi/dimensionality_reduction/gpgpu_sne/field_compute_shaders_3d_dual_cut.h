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

#include "hdi/dimensionality_reduction/gpgpu_sne/utils/verbatim.h"

GLSL(bvh_dual_dispatch_src, 450,
  layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  layout(binding = 7, std430) restrict readonly buffer Value { uint value; };
  layout(binding = 14, std430) restrict writeonly buffer Disp { uvec3 dispatch; };
  layout(location = 0) uniform uint div;

  void main() {
    dispatch = uvec3((value + div - 1) / div, 1, 1);
  }
);

GLSL(field_bvh_dual_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_shuffle : require )   // subgroupShuffle(...) support

  // Wrapper structure for pair queue data
  struct Pair {
    uint v;       // View BVH node index
    uint p;       // Point BVH node index
  };
  
  // Wrapper structure for tree node data
  struct Node {
    vec3 center;  // Center of mass or bounding box
    uint begin;   // Begin of range of contained data
    vec3 diam;    // Diameter of bounding box
    uint extent;  // Extent of range of contained data
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer PNod0 { vec4 pNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer PNod1 { vec4 pNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer PPosi { vec3 pPosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer VNod0 { vec4 vNode0Buffer[]; };
  layout(binding = 4, std430) restrict readonly buffer VNod1 { vec4 vNode1Buffer[]; };
  layout(binding = 5, std430) restrict coherent buffer VFiel { float vFieldBuffer[]; };  // Atomic
  layout(binding = 6, std430) restrict readonly buffer IQueu { Pair iQueueBuffer[]; };   // Input
  layout(binding = 7, std430) restrict readonly buffer IQHea { uint iQueueHead; };       // Atomic
  layout(binding = 8, std430) restrict writeonly buffer OQue { Pair oQueueBuffer[]; };   // Output
  layout(binding = 9, std430) restrict readonly buffer OQHea { uint oQueueHead; };       // Atomic
  layout(binding = 10, std430) restrict writeonly buffer LQueu { Pair lQueueBuffer[]; }; // Leaf
  layout(binding = 11, std430) restrict coherent buffer LQHea { uint lQueueHead; };      // Atomic
  layout(binding = 12, std430) restrict writeonly buffer AQueu { Pair aQueueBuffer[]; }; // Accept
  layout(binding = 13, std430) restrict coherent buffer AQHea { uint aQueueHead; };      // Atomic

  // Uniform values
  layout(location = 0) uniform uint nPointLvls;  // Nr. of levels in point BVH
  layout(location = 1) uniform uint nViewLvls;   // Nr. of levels in view BVH
  layout(location = 2) uniform uint kNode;       // Node fanout in point/view BVH
  layout(location = 3) uniform uint kLeaf;       // Leaf fanout in point BVH
  layout(location = 4) uniform float theta2;     // Squared approximation param. for Barnes-Hut
  layout(location = 5) uniform bool pushAcceptQueue;
  
  // Constant values
  const uint logk = uint(log2(kNode));
  const uint thread = gl_LocalInvocationID.x % kNode;

  float sdot(in vec3 v) {
    return dot(v, v);
  }

  void main() {
    // Each kNode invocations share a single node pair to subdivide
    const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / kNode;
    if (i >= iQueueHead) {
      return;
    }
    
    // Read node pair indices using subgroup operations
    Pair pair;
    if (thread == 0) {
      pair = iQueueBuffer[i];
    }
    pair.v = subgroupShuffle(pair.v, gl_LocalInvocationID.x - thread);
    pair.p = subgroupShuffle(pair.p, gl_LocalInvocationID.x - thread);

    // Read node pair data using subgroup operations
    const vec4 vNode0 = subgroupShuffle((thread == 0) 
      ? vNode0Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 vNode1 = subgroupShuffle((thread == 1) 
      ? vNode1Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    const vec4 pNode0 = subgroupShuffle((thread == 2) 
      ? pNode0Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const vec4 pNode1 = subgroupShuffle((thread == 3) 
      ? pNode1Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x + 3 - thread);
    Node pNode = Node(pNode0.xyz, uint(pNode1.w), pNode1.xyz, uint(pNode0.w));
    Node vNode = Node(vNode0.xyz + 0.5 * vNode1.xyz, uint(vNode1.w),  vNode1.xyz, uint(vNode0.w));

    // Determine node levels
    uint pLvl = 0;
    uint vLvl = 0;
    {
      uint p = pair.p;
      uint v = pair.v;
      do { pLvl++; } while ((p = (p - 1) >> logk) > 0);
      do { vLvl++; } while ((v = (v - 1) >> logk) > 0);
    }

    // Test whether either of these nodes can subdivide (alternatively, one would be a leaf node)
    bool pCanSubdiv = pLvl < nPointLvls - 1 && pNode.extent > kLeaf;
    bool vCanSubdiv = vLvl < nViewLvls - 1 && vNode.extent > 1;

    // Perform subdivision on view or point tree
    // Decision is based on (a) is anyone a leaf already and (b) which bbox has the larger diameter
    const bool vDidSubdiv = (vCanSubdiv && sdot(vNode.diam) > sdot(pNode.diam)) || !pCanSubdiv;
    if (vDidSubdiv) {
      // Step down in view tree one level, offset thread for fanout
      pair.v = pair.v * kNode + 1 + thread;

      // Load new view node
      const vec4 _vNode0 = vNode0Buffer[pair.v]; // bbox minima, range extent
      const vec4 _vNode1 = vNode1Buffer[pair.v]; // bbox extent, range min
      vNode = Node(_vNode0.xyz + 0.5 * _vNode1.xyz, uint(_vNode1.w),  _vNode1.xyz, uint(_vNode0.w));

      // Test if this node can still subdivide (alternatively, it is a leaf node)
      vCanSubdiv = ++vLvl < nViewLvls - 1 && vNode.extent > 1;
    } else {
      // Step down in point tree one level, offset thread for fanout
      pair.p = pair.p * kNode + 1 + thread;

      // Load new point node
      const vec4 _pNode0 = pNode0Buffer[pair.p]; // bbox center, range extent
      const vec4 _pNode1 = pNode1Buffer[pair.p]; // bbox extent, range min
      pNode = Node(_pNode0.xyz, uint(_pNode1.w), _pNode1.xyz, uint(_pNode0.w));

      // Test if this node can still subdivide (alternatively, it is a leaf node)
      pCanSubdiv = ++pLvl < nPointLvls - 1 && pNode.extent > kLeaf;
    }

    // Test for a dead end in the tree to avoid inaccuracies when computing forces
    vec4 field = vec4(0);
    if (pNode.extent != 0 && vNode.extent != 0) {
      // Compute squared distance between point and view nodes
      const vec3 t = vNode.center - pNode.center;
      const float t2 = dot(t, t);

      // Align the distance vector so it is always coming from the same part of the axis
      // such that the largest possible diameter of the bounding box is visible
      const vec3 b = abs(t) * vec3(-1, -1, 1);
      const vec3 b_ = normalize(b);

      // Compute largest squared diameter, adjusted for viewing angle, for point and view nodes
      // This is the vector rejection of diam onto unit vector b_
      const float r2 = max(
        sdot(pNode.diam - b_ * dot(pNode.diam, b_)), // Vector rejection of b onto p's vector
        sdot(vNode.diam - b_ * dot(vNode.diam, b_))  // Vector rejection of b onto v's vector
      );

      // Dual-tree Barnes-Hut approximation is currently based on the
      // formulation in "Accelerating t-SNE using Tree-Based Algorithms", 
      // L. van der Maaten, 2014. Src: http://jmlr.org/papers/v15/vandermaaten14a.html
      // but with a better fitting r parameter computed above
      if (r2 / t2 < theta2) {
        // Approximation passes, compute approximated value and truncate
        const float tStud = 1.f / (1.f + t2);
        field = pNode.extent * vec4(tStud, t * (tStud * tStud));
      } else if (!vCanSubdiv && !pCanSubdiv) {
        // Leaf is reached. Large leaves are dealt with in separate shader
        // as leaves require iterating over all contained data
        if (pNode.extent > 16) {
          lQueueBuffer[atomicAdd(lQueueHead, 1)] = pair;
        } else {
          for (uint j = pNode.begin; j < pNode.begin + pNode.extent; ++j) {
            const vec3 t = vNode.center - pPosBuffer[j];
            const float tStud = 1.f / (1.f +  dot(t, t));
            field += vec4(tStud, t * (tStud * tStud));
          }
        }
      } else {
        // Push pair on work queue for further subdivision
        oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
      }
    } 

    // Add computed forces to field
    if (vDidSubdiv && field != vec4(0)) {
      // When the view tree is subdivided, each invocation must do its own writes
      const uvec4 addr = uvec4(4 * pair.v) + uvec4(0, 1, 2, 3);
      atomicAdd(vFieldBuffer[addr.x], field.x);
      atomicAdd(vFieldBuffer[addr.y], field.y);
      atomicAdd(vFieldBuffer[addr.z], field.z);
      atomicAdd(vFieldBuffer[addr.w], field.w);
    } else if (!vDidSubdiv) {
      // All threads can enter this so the subgroup can be used without inactive invocations
      // When the point tree is subdivided, kNode invocations can write to the same view node
      field = subgroupClusteredAdd(field, kNode);
      if (thread < 4 && field != vec4(0)) {
        atomicAdd(vFieldBuffer[4 * pair.v + thread], field[thread]);
      }
    }

    // Push pair on accept queue to store intermediate cut of dual tree
    if (pushAcceptQueue && field != vec4(0)) {
      aQueueBuffer[atomicAdd(aQueueHead, 1)] = pair;
    }
  }
);

GLSL(field_bvh_leaf_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_shuffle : require )   // subgroupShuffle(...) support
  
  // Wrapper structure for pair queue data
  struct Pair {
    uint v; // View BVH node index
    uint p; // Point BVH node index
  };

  // Wrapper structure for node data
  struct Node {
    vec3 center; 
    uint begin;
    vec3 diam;
    uint extent;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
  
  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer PNod0 { vec4 pNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer PNod1 { vec4 pNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer PPosi { vec3 pPosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer VNod0 { vec4 vNode0Buffer[]; };
  layout(binding = 4, std430) restrict readonly buffer VNod1 { vec4 vNode1Buffer[]; };
  layout(binding = 5, std430) restrict coherent buffer VFiel { float vFieldBuffer[]; }; // atomic rw
  layout(binding = 6, std430) restrict buffer LQueu { Pair lQueueBuffer[]; };
  layout(binding = 7, std430) restrict coherent buffer LQHea { uint lQueueHead; }; // atomic rw 
  
  // Constants
  const uint nThreads = 4; // Two threads to handle all the buffer reads together
  const uint thread = gl_LocalInvocationID.x % nThreads;

  float sdot(vec3 v) {
    return dot(v, v);
  }

  void main() {
    // Check if invoc is within nr of items on accept queue
    uint j = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
    if (j >= lQueueHead) {
      return;
    }

    // Load node pair from queue, skip culled pairs
    Pair pair = lQueueBuffer[j];

    // Read node data using subgroup operations
    const vec4 vNode0 = subgroupShuffle((thread == 0) 
      ? vNode0Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 vNode1 = subgroupShuffle((thread == 1) 
      ? vNode1Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    const vec4 pNode0 = subgroupShuffle((thread == 2) 
      ? pNode0Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x + 2 - thread);
    const vec4 pNode1 = subgroupShuffle((thread == 3) 
      ? pNode1Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x + 3 - thread);
    const Node pNode = Node(pNode0.xyz, uint(pNode1.w), pNode1.xyz, uint(pNode0.w));
    const Node vNode = Node(vNode0.xyz + 0.5 * vNode1.xyz, uint(vNode1.w),  vNode1.xyz, uint(vNode0.w));

    // Compute forces for this node pair directly
    vec4 field = vec4(0);
    for (uint i = pNode.begin + thread; 
              i < pNode.begin + pNode.extent; 
              i += nThreads) {
      vec3 t = vNode.center - pPosBuffer[i];
      float tStud = 1.f / (1.f +  dot(t, t));
      field += vec4(tStud, t * (tStud * tStud));
    }
    field = subgroupClusteredAdd(field, nThreads);

    // Add computed forces to field
    if (field != vec4(0)) {
      const uvec4 addr = uvec4(4 * pair.v) + uvec4(0, 1, 2, 3);
      atomicAdd(vFieldBuffer[addr[thread]], field[thread]);
    }
  }
);

GLSL(iterate_src, 450,
  GLSL_PROTECT( #extension GL_NV_shader_atomic_float : require )        // atomicAdd(f32) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_clustered : require ) // subgroupClusteredAdd(...) support
  GLSL_PROTECT( #extension GL_KHR_shader_subgroup_shuffle : require )   // subgroupShuffle(...) support

  // Return types for approx(...) function below
  #define DEAD 0\n // Empty node(s), truncate early
  #define APPR 1\n // Test passes (BH), approximate forces
  #define LEAF 2\n // Test passes (leaf), compute forces exactly
  #define FAIL 3\n // Test fails, subdivide

  // Wrapper structure for pair queue data
  struct Pair {
    uint v; // View BVH node index
    uint p; // Point BVH node index
  };
  
  // Wrapper structure for tree node data
  struct Node {
    vec3 center; 
    uint begin;
    vec3 diam;
    uint extent;
  };

  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer PNod0 { vec4 pNode0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer PNod1 { vec4 pNode1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer PPosi { vec3 pPosBuffer[]; };
  layout(binding = 3, std430) restrict readonly buffer VNod0 { vec4 vNode0Buffer[]; };
  layout(binding = 4, std430) restrict readonly buffer VNod1 { vec4 vNode1Buffer[]; };
  layout(binding = 5, std430) restrict coherent buffer VFiel { float vFieldBuffer[]; }; // atomic rw
  layout(binding = 6, std430) restrict buffer AQueu { Pair aQueueBuffer[]; };
  layout(binding = 7, std430) restrict coherent buffer AQHea { uint aQueueHead; }; // atomic rw 
  
  // Uniform values
  layout(location = 0) uniform uint nPointLvls;  // Nr. of levels in point BVH
  layout(location = 1) uniform uint nViewLvls;   // Nr. of levels in view BVH
  layout(location = 2) uniform uint kNode;       // Node fanout in point/view BVH
  layout(location = 3) uniform uint kLeaf;       // Leaf fanout in point BVH
  layout(location = 4) uniform float theta2;     // Squared approximation param. for Barnes-Hut
  
  // Constant values
  const uint logk = uint(log2(kNode));
  const float invLogk = 1.f / (float(logk));
  const uint nThreads = 2; // Two threads to handle the buffer reads together
  const uint thread = gl_LocalInvocationID.x % nThreads;

  float sdot(vec3 v) {
    return dot(v, v);
  }

  uint approx(Node v, uint vLvl, Node p, uint pLvl) {
    // Truncate early if a node is empty
    if (p.extent == 0 || v.extent == 0) {
      return DEAD; 
    } 

    // Radius, adjusted for viewing angle, for point and view node bounding spheres
    const vec3 t = v.center - p.center;
    const vec3 b = abs(normalize(t)) * vec3(-1, -1, 1);
    const float r2 = max(
       sdot(p.diam - b * dot(p.diam, b)), // Vector rejection of b onto p.diam
       sdot(v.diam - b * dot(v.diam, b))  // Vector rejection of b onto v.diam
    );
    
    if (r2 / sdot(t) < theta2) {
      return APPR; // BH Approximation passes, forces should be added
    } else if ((p.extent <= kLeaf || pLvl == nPointLvls - 1)
            && (v.extent <= 1 || vLvl == nViewLvls - 1)) {
      return LEAF; // Is leaf, forces should be added
    } else {
      return FAIL; // Would have to be subdivided
    }
  }

  void main() {
    // Check if invoc is within nr of items on accept queue
    uint j = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
    if (j >= aQueueHead) {
      return;
    }

    // Load node pair from queue, skip culled pairs
    Pair pair = aQueueBuffer[j];
    if (pair.p == 0u || pair.v == 0u) {
      return;
    }

    // Read node data using subgroup operations
    const vec4 vNode0 = subgroupShuffle((thread == 0) 
      ? vNode0Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 vNode1 = subgroupShuffle((thread == 1) 
      ? vNode1Buffer[pair.v] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    const vec4 pNode0 = subgroupShuffle((thread == 0) 
      ? pNode0Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x - thread);
    const vec4 pNode1 = subgroupShuffle((thread == 1) 
      ? pNode1Buffer[pair.p] : vec4(0),
      gl_LocalInvocationID.x + 1 - thread);
    Node pNode = Node(pNode0.xyz, uint(pNode1.w), pNode1.xyz, uint(pNode0.w));
    Node vNode = Node(vNode0.xyz + 0.5 * vNode1.xyz, uint(vNode1.w),  vNode1.xyz, uint(vNode0.w));

    // Determine node levels
    uint pLvl = 0;
    uint vLvl = 0;
    {
      uint p = pair.p;
      uint v = pair.v;
      do { pLvl++; } while ((p = (p - 1) >> logk) > 0);
      do { vLvl++; } while ((v = (v - 1) >> logk) > 0);
    }

    // Inverse subdivide to find a suitable parent node pair, and read that using subgroup ops
    const bool pChild = sdot(pNode.diam) < sdot(vNode.diam) && pLvl > 0;
    Pair _pair = pair;
    Node _pNode = pNode;
    Node _vNode = vNode;
    if (pChild) {
      _pair.p = (_pair.p - 1) >> logk;
      const vec4 _pNode0 = subgroupShuffle((thread == 0) 
        ? pNode0Buffer[_pair.p] : vec4(0),
        gl_LocalInvocationID.x - thread);
      const vec4 _pNode1 = subgroupShuffle((thread == 1) 
        ? pNode1Buffer[_pair.p] : vec4(0),
        gl_LocalInvocationID.x + 1 - thread);
      _pNode = Node(_pNode0.xyz, uint(_pNode1.w), _pNode1.xyz, uint(_pNode0.w));
    } else {
      _pair.v = (_pair.v - 1) >> logk;
      const vec4 _vNode0 = subgroupShuffle((thread == 0) 
        ? vNode0Buffer[_pair.v] : vec4(0),
        gl_LocalInvocationID.x - thread);
      const vec4 _vNode1 = subgroupShuffle((thread == 1) 
        ? vNode1Buffer[_pair.v] : vec4(0),
        gl_LocalInvocationID.x + 1 - thread);
      _vNode = Node(_vNode0.xyz + 0.5 * _vNode1.xyz, uint(_vNode1.w), _vNode1.xyz, uint(_vNode0.w));
    }

    // Test both pairs with barnes-hut approx
    const uint grappr = approx(thread == 0 ? vNode : _vNode, thread == 0 ? vLvl : 0,
                               thread == 0 ? pNode : _pNode, thread == 0 ? pLvl : 0);
    uint appr = subgroupShuffle(grappr, gl_LocalInvocationID.x - thread);
    uint _appr = subgroupShuffle(grappr, gl_LocalInvocationID.x - thread + 1);

    /*
      Results of test imply one of the following
      1. Both dead
      2. Parent approx or leaf node
      3. Child not approx or leaf node
      4. Neither approx :shrug:
    */

    if (appr == DEAD && _appr == DEAD) { // Something dead was encountered, just exit
      return;
    } else if (_appr != FAIL) { // Parent passes test or is leaf, move up tree
      // Is invoc the leftmost child node?
      const bool leftChild = (pChild ? pair.p : pair.v) % kNode == 0;

      // If leftmost child, move to parent and carry on
      // All other children set their pair as invalid and return
      pair = leftChild ? _pair : Pair(0, 0);
      if (thread == 0) {
        aQueueBuffer[j] = pair;
      }
      if (!leftChild) {
        return;
      }
      
      appr = _appr;
      vNode = _vNode;
      pNode = _pNode;
    } else if (appr == FAIL) { // Current node fails test, as does parent, so move down tree
      // Perform subdivision on either view or point tree depending on circumstances
      const bool vSubdiv =
      !(pLvl < nPointLvls - 1 && pNode.extent > kLeaf) ||
       (vLvl < nViewLvls - 1 && vNode.extent > 1 && sdot(vNode.diam) > sdot(pNode.diam));

      // Add new subdivided pairs to queue
      uint head = 0;
      if (thread == 0) {
        head = atomicAdd(aQueueHead, kNode);
        aQueueBuffer[j] = Pair(0, 0); // Flag current pair as invalid (can't exactly remove it...)
      }
      head = subgroupShuffle(head, gl_LocalInvocationID.x - thread);
      for (uint i = thread; i < kNode; i += nThreads) {
        aQueueBuffer[head + i] = Pair(
          vSubdiv ? (pair.v * kNode) + 1u + i : pair.v,
          vSubdiv ? pair.p : (pair.p * kNode) + 1u + i
        );
      }
      return;
    }

    // Compute forces for this node pair
    vec4 field = vec4(0);
    if (appr == APPR) {
      // If BH-approximation passes, compute approximated value and truncate
      vec3 t = vNode.center - pNode.center;
      float t2 = dot(t, t);
      float tStud = 1.f / (1.f + t2);
      field = pNode.extent * vec4(tStud, t * (tStud * tStud));
    } else {
      // Leaf node, iterate over all points and add to view node
      for (uint i = pNode.begin + thread; i < pNode.begin + pNode.extent; i += nThreads) {
        vec3 t = vNode.center - pPosBuffer[i];
        float tStud = 1.f / (1.f +  dot(t, t));
        field += vec4(tStud, t * (tStud * tStud));
      }
      field = subgroupClusteredAdd(field, nThreads);
    }
    
    // Add computed forces to field
    if (field != vec4(0)) {
      const uvec4 addr = uvec4(4 * pair.v) + uvec4(0, 1, 2, 3);
      atomicAdd(vFieldBuffer[addr[2 * thread]], field[2 * thread]);
      atomicAdd(vFieldBuffer[addr[2 * thread + 1]], field[2 * thread + 1]);
    }
  }
);

GLSL(push_src, 450,
  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

  // Buffer bindings
  layout(binding = 0, std430) restrict readonly buffer VNod0 { vec4 node0Buffer[]; };
  layout(binding = 1, std430) restrict readonly buffer VNod1 { vec4 node1Buffer[]; };
  layout(binding = 2, std430) restrict readonly buffer VPixe { uvec3 posBuffer[]; };
  layout(binding = 3, std430) restrict coherent buffer VFiel { vec4 fieldBuffer[]; };
  layout(binding = 4, std430) restrict readonly buffer LFlag { uint leafBuffer[]; };
  layout(binding = 5, std430) restrict readonly buffer LHead { uint leafHead; };

  // Image bindings
  layout(binding = 0, rgba32f) restrict writeonly uniform image3D fieldImage;

  // Uniform values
  layout(location = 0) uniform uint kNode; // Node fanout in BVH
  
  // Constant values
  const uint logk = uint(log2(kNode));

  void main() {
    // Check if invoc is within nr of items on leaf queue
    const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    if (j >= leafHead) {
      return;
    }

    const uint i = leafBuffer[j];
    
    // Push forces down by ascending up tree to root
    vec4 field = vec4(0); // fieldBuffer[0]; // ignore root, it will never approximate
    for (int k = int(i); k > 0; k = (k - 1) >> logk) {
      field += fieldBuffer[k];
    }

    // Store resulting field value in underlying image pixel(s)
    const uint begin = uint(node1Buffer[i].w);
    const uint extent = uint(node0Buffer[i].w);
    for (uint k = begin; k < begin + extent; k++) {
      imageStore(fieldImage, ivec3(posBuffer[k]), field);
    }
  }
);