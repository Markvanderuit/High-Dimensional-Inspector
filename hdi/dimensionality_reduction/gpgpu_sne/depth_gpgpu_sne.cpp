#ifndef __APPLE__

#include "depth_gpgpu_sne.h"

namespace {
  void glAssert(const std::string& msg) {
    GLenum err; 
    while ((err = glGetError()) != GL_NO_ERROR) {
      printf("Location: %s, err: %i\n", msg.c_str(), err);
    }
  }

  enum BufferType {
    POSITION,
    INTERP_FIELDS
    // ...
  };
}

namespace  hdi::dr {
  GpgpuDepthSneCompute::GpgpuDepthSneCompute() {
    // ...
  }
  
  void GpgpuDepthSneCompute::clean() {
    // ...
  }

  void GpgpuDepthSneCompute::initialize() {
    // ...
  }

  void GpgpuDepthSneCompute::compute() {
    // ...
  }
}


#endif // __APPLE__