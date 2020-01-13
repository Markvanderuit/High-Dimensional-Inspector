#pragma once
#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

// Vertex shader for near/far depth map computation
GLSL(depth_vert_src, 430,
  layout(location = 0) in vec4 point;
  layout(std430, binding = 0) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  uniform mat4 transformation;

  void main() {
    // Transform point into [0, 1] space
    vec3 position = (point.xyz - minBounds) * invRange;

    // Apply camera transformation
    position = vec3(transformation * vec4(position, 1));
    
    // Transform point into [-1, 1] clip space
    gl_Position =  vec4(position * 2.f - 1.f, 1.f);
  }
);

// Geometry shader for near/far depth map computation
GLSL(depth_geom_src, 430,
  layout (points) in;
  layout (points, max_vertices = 1) out;

  void main() {
    // Emit point for front depth boundary
    gl_Layer = 0;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();

    // Emit point for back depth boundary
    gl_Layer = 1;
    gl_Position = gl_in[0].gl_Position * vec4(1, 1, -1, 1);
    EmitVertex();
    EndPrimitive();
  }
);

// Fragment shader for voxel grid computation and depth map formatting
GLSL(grid_fragment_src, 430,
  // uvec4 gives us 32 32 32 32 = 128 z values in grid, in 1 rendertarget
  layout (location = 0) out uvec4 color;
  // xyzw = zNear, zFar, zRange, invZRange
  layout (location = 1) out vec4 depth;

  uniform sampler2DArray depthMaps;
  uniform usampler1D cellMap;
  uniform float zPadding;

  void main() {
    vec2 xyFixed = (gl_FragCoord.xy) / textureSize(depthMaps, 0).xy;

    // zFar still has to be reversed
    float zNear = texture(depthMaps, vec3(xyFixed, 0), 0).x;
    float zFar = 1.f - texture(depthMaps, vec3(xyFixed, 1), 0).x;

    // Add padding to stimulate growth of embeddnig bounds
    zNear -= zPadding;
    zFar += zPadding;

    // Store these values as we get 3/4th free texture lookups
    float zRange = zFar - zNear;
    float zRangeDiv = 1.f / zRange;
    depth = vec4(zNear, zFar, zRange, zRangeDiv); // x y z w to store these four values

    // Store lookup of cell map for this particular z bound by zNear and zFar
    // Compute z bounded by [zNear, zFar], convert to cellmap texture resolution
    float zUnfixed = (gl_FragCoord.z - zNear) * zRangeDiv;
    float zFixed = zUnfixed * textureSize(cellMap, 0).x - 0.5;

    // Store OR of the two cells as result
    uvec4 lowerCell = texelFetch(cellMap, int(floor(zFixed)), 0);
    uvec4 greaterCell = texelFetch(cellMap, int(ceil(zFixed)), 0);
    color = lowerCell | greaterCell;
  }
);

// Compute shader for field computation
GLSL(field_src, 430,
  layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer PosInterface { vec3 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };
  layout(rgba32f, binding = 0) writeonly uniform image3D fields_texture;
  
  uniform uint num_points;
  uniform uvec3 texture_size;
  uniform mat4 invTransformation;
  uniform sampler2D depth_texture;
  uniform usampler1D cell_texture;
  uniform usampler2D grid_texture;

  // Reduction components
  const uint groupSize = gl_WorkGroupSize.x;
  const uint halfGroupSize = groupSize / 2;
  shared vec4 reductionArray[halfGroupSize];

  void main() {
    // Location of current workgroup
    ivec2 xyFixed = ivec2(gl_WorkGroupID.xy);
    vec2 xyUnfixed = (vec2(xyFixed) + vec2(0.5)) / vec2(texture_size.xy);
    uint lid = gl_LocalInvocationIndex.x;

    // Query grid map
    uvec4 gridVec = texture(grid_texture, xyUnfixed);

    // If grid value is all zeroes, we can just skip z-row of pixels entirely
    if (gridVec == uvec4(0)) {
      uint z = gl_WorkGroupID.z * gl_WorkGroupSize.z + lid;
      if (z < texture_size.z) {
        imageStore(fields_texture, ivec3(xyFixed, z), vec4(0));
      }
      return;
    }

    // Query depth map
    vec4 zValues = texture(depth_texture, xyUnfixed);

    for (uint z = gl_WorkGroupID.z; 
          z < texture_size.z; 
          z += gl_NumWorkGroups.z) {
      // Map xyz to [0, 1], actually bound by [zNear, zFar] though
      ivec3 xyzFixed = ivec3(xyFixed, z);
      vec3 domain_pos = (vec3(xyzFixed) + vec3(0.5)) / vec3(texture_size);

      // Query grid map sample or skip pixel
      if ((gridVec & texture(cell_texture, domain_pos.z)) == uvec4(0)) {
        if (lid < 1) {
          imageStore(fields_texture, xyzFixed, vec4(0));
        }
        continue;
      }

      // Map z from [zNear, zFar] to [0, 1] using queried depth
      domain_pos.z = domain_pos.z * zValues.z + zValues.x;

      // Undo camera transformation
      domain_pos = vec3(invTransformation * vec4(domain_pos, 1));

      // Map to domain bounds
      domain_pos = domain_pos * range + minBounds;

      // Iterate over points to obtain density/gradient
      vec4 v = vec4(0);
      for (uint i = lid; i < num_points; i += groupSize) {
        vec3 t = domain_pos - Positions[i];
        float t_stud = 1.f / (1.f + dot(t, t));
        vec3 t_stud_2 = t * (t_stud * t_stud);

        // Field layout is: S, V.x, V.y, V.z
        v += vec4(t_stud, t_stud_2);
      }
      
      // Perform reduce add over all computed points for this pixel
      if (lid >= halfGroupSize) {
        reductionArray[lid - halfGroupSize] = v;
      }
      barrier();
      if (lid < halfGroupSize) {
        reductionArray[lid] += v;
      }
      for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
        barrier();
        if (lid < i) {
          reductionArray[lid] += reductionArray[lid + i];
        }
      }
      barrier();
      if (lid < 1) {
        vec4 reducedArray = reductionArray[0] + reductionArray[1];
        imageStore(fields_texture, xyzFixed, reducedArray);
      }
      barrier();
    }
  }
);

// Compute shader for point sampling from field
GLSL(interp_src, 430,
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
  layout(std430, binding = 0) buffer Pos{ vec3 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer BoundsInterface { 
    vec3 minBounds;
    vec3 maxBounds;
    vec3 range;
    vec3 invRange;
  };

  uniform uint num_points;
  uniform uvec3 texture_size;
  uniform mat4 transformation;
  uniform sampler2D depth_texture;
  uniform sampler3D fields_texture;

  vec4 sampleAt(float zUnfixed, ivec2 positionFixed) {
    // Map zUnfixed to [zNear, zFar] for this specific x,y position
    vec4 zValues = texelFetch(depth_texture, positionFixed, 0);
    float z = clamp((zUnfixed - zValues.x) * zValues.w, 0.f, 1.f);

    // Map z to [0, texture size]
    float zFixed = z * float(texture_size.z) - 0.5;

    // Query field texture at fixed z-positions
    vec4 fieldFront = texelFetch(fields_texture, ivec3(positionFixed, floor(zFixed)), 0);
    vec4 fieldBack = texelFetch(fields_texture, ivec3(positionFixed, ceil(zFixed)), 0);

    // Interpolate queried field values
    float delta = zFixed - floor(zFixed);
    return mix(fieldFront, fieldBack, delta);
  }

  void main() {
    // Grid stride loop, straight from CUDA, scales better for very large N
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
        i < num_points;
        i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      // Map position of point to [0, 1]
      vec3 position = (Positions[i] - minBounds) * invRange;
      
      // Apply optional camera transformation
      position = vec3(transformation * vec4(position, 1));

      // Locations for sampling field at four fixed positions
      vec2 positionFixed = position.xy * vec2(texture_size.xy) - vec2(0.5);
      vec2 positionFloor = floor(positionFixed);
      vec2 positionCeil = ceil(positionFixed);
      vec2 delta = positionFixed - positionFloor;

      // Interpolate the four sampled positions
      Values[i] = mix(mix(sampleAt(position.z, ivec2(positionFloor)),
                          sampleAt(position.z, ivec2(positionCeil.x, positionFloor.y)),
                          delta.x), 
                      mix(sampleAt(position.z, ivec2(positionFloor.x, positionCeil.y)),
                          sampleAt(position.z, ivec2(positionCeil)),
                          delta.x), 
                      delta.y);
    }
  }
);