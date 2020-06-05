#pragma once

#define GLSL(name, version, shader) static const char *name = "#version " #version "\n" #shader

GLSL(reduce_src, 430,
     struct ReduceIterOut {
       vec3 min;
       vec3 max;
     };

     layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
     layout(binding = 0, std430) restrict buffer Iter0Buffer { ReduceIterOut reduceIter0Out[]; };
     layout(binding = 1, std430)
         restrict writeonly buffer Iter1Buffer { ReduceIterOut reduceIter1Out; };
     layout(location = 0) uniform sampler2D textureSampler;
     layout(location = 1) uniform ivec2 textureSize; layout(location = 2) uniform uint iteration;
     layout(location = 3) uniform uint reduceIter0Size;

     // Local and global invocation indices
     const ivec2 gxy = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy);
     const uint gid = gl_WorkGroupSize.x * gl_WorkGroupID.y + gl_WorkGroupID.x;
     const uint lid = gl_LocalInvocationIndex;

     // Shared memory spans half the workgroup
     const uint groupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
     const uint halfGroupSize = groupSize / 2; shared ReduceIterOut reduceShared[halfGroupSize];

     void main() {
       // Only keep invocations inside image active
       if (iteration == 0 && min(gxy, textureSize - 1) != gxy) {
         return;
       }

       ReduceIterOut v = ReduceIterOut(vec3(9999), vec3(0));
       if (iteration == 0) {
         // Read current image value
         v.max = texelFetch(textureSampler, gxy, 0).xyz;
         v.min = v.max;
       } else {
         // Read number of values from previous reduction
         for (uint i = lid; i < reduceIter0Size; i += groupSize) {
           v.max = max(v.max, reduceIter0Out[i].max);
           v.min = min(v.min, reduceIter0Out[i].min);
         }
       }

       // Perform parallel reduction
       if (lid >= halfGroupSize) {
         reduceShared[lid - halfGroupSize] = v;
       }
       barrier();
       if (lid < halfGroupSize) {
         reduceShared[lid].max = max(reduceShared[lid].max, v.max);
         reduceShared[lid].min = min(reduceShared[lid].min, v.min);
       }
       for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
         barrier();
         if (lid < i) {
           reduceShared[lid].max = max(reduceShared[lid].max, reduceShared[lid + i].max);
           reduceShared[lid].min = min(reduceShared[lid].min, reduceShared[lid + i].min);
         }
       }
       barrier();

       // Store result depending on iteration
       if (lid < 1) {
         v.max = max(reduceShared[0].max, reduceShared[1].max);
         v.min = min(reduceShared[0].min, reduceShared[1].min);
         if (iteration == 0) {
           reduceIter0Out[gid] = v;
         } else {
           reduceIter1Out = v;
         }
       }
     });

GLSL(normalize_src, 430,
     struct Bounds {
       vec3 min;
       vec3 max;
     };

     layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
     layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
     layout(binding = 0, rgba32f) restrict writeonly uniform image2D normalizedImage;
     layout(location = 0) uniform sampler2D textureSampler;
     layout(location = 1) uniform ivec2 textureSize;

     // Local and global invocation indices
     const ivec2 gxy = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy);
     const uint lid = gl_LocalInvocationIndex;

     // Shared access of buffer value
     shared vec3 minShared; shared vec3 invRangeShared;

     void main() {
       // Only keep invocations active inside image boundary
       if (min(gxy, textureSize - 1) != gxy) {
         return;
       }

       // Load bounds values from buffer into shared memory
       if (lid == 0) {
         minShared = bounds.min;
         invRangeShared = 1.f / (bounds.max - bounds.min);
       }
       barrier();

       // Fetch texture value, normalize it, write it to image
       vec3 v = texelFetch(textureSampler, gxy, 0).xyz;
      //  if (v != vec3(0)) {
      //    v = (v - minShared) * invRangeShared;
      //  } else {
      //    v = vec3(1);
      //  }
       imageStore(normalizedImage, gxy, vec4(v, 0));
     });

GLSL(screen_vert_src, 430, layout(location = 0) in vec2 vertexPosition;
     layout(location = 0) out vec2 texelPosition;

     void main() {
       texelPosition = 0.5f * vertexPosition + 0.5f;
       gl_Position = vec4(vertexPosition, 0, 1);
     });

GLSL(screen_frag_src, 430, layout(location = 0) in vec2 texelPosition;
     layout(location = 0) out vec3 color;
     layout(location = 0) uniform sampler2D textureSampler;

    //  layout(location = 0) out vec4 color;
    //  layout(location = 0) uniform usampler2D textureSampler;

     void main() {
       // Visualize voxel grid using bitCount() / 32.f
      //  uvec4 v = texture(textureSampler, texelPosition);
      //  ivec4 density = bitCount(v);
      //  color = 1 - vec4(density) / 32.f;

       // Visualise density/gradient field
       color = texture(textureSampler, texelPosition).xyz;
     });