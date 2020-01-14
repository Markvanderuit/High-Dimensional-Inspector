#pragma once

namespace hdi::dr {
  // Simple 3D point representation
  // vec3 gets to vec4 in std140/430, so we align to 16 bytes
  struct alignas(16) Point3D {
    float x, y, z;
  };

  // Simple 3D bounds representation
  struct Bounds3D {
    Point3D min, max;

    Point3D range() {
      return Point3D {
        max.x - min.x,
        max.y - min.y,
        max.z - min.z
      };
    }

    Point3D center() {
      return Point3D {
        0.5f * (min.x + max.x),
        0.5f * (min.y + max.y),
        0.5f * (min.z + max.z), 
      };
    }
  };
};