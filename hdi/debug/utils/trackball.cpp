#include "hdi/debug/utils/trackball.hpp"
#include <glm/gtx/transform.hpp>
#include <GLFW/glfw3.h>
#include <iostream>

/**
 * Trackball math based on:
 * - https://github.com/sanko/fltk-2.0.x/blob/master/test/trackball.c
 * and
 * - https://www.khronos.org/opengl/wiki/Object_Mouse_Trackball
 */

float projectToSphere(float r, glm::vec2 pos) {
  float d = glm::length(pos);
  if (d < r * 0.70710678118654752440) {
    // Inside sphere code
    return glm::sqrt(r * r - d * d);
  } else {
    // On hyperbola code
    float t = r / 1.41421356237309504880;
    return t * t / d;
  }
}

namespace hdi::dbg {
  Trackball::Trackball()
  : _r(1),
    _s(2.25),
    _m(1),
    _isTracking(false) 
  { }

  void Trackball::compute()
  { 
    if (_isTracking && _prevPos != _currPos) {
      // Compute simplified plane rotation
      glm::vec2 v = _currPos - _prevPos;
      glm::mat4 rx = glm::rotate(v.x, glm::vec3(0, 1, 0));
      glm::mat4 ry = glm::rotate(v.y, glm::vec3(1, 0, 0));

      // Update lookat matrix
      _r = ry * rx * _r;
    }

    _m = glm::lookAt(_s * glm::vec3(0, 0, 1), glm::vec3(0), glm::vec3(0, 1, 0)) * _r;
  }

  glm::mat4 Trackball::matrix() const {
    return _m;
  }

  void Trackball::mousePosInput(double xPos, double yPos)
  {
    auto *ptr = hdi::dbg::Window::currentWindow();
    
    // Record previous position as last current position
    _prevPos = _currPos;

    // Record current position in space [-1, 1]
    _currPos = glm::vec2(xPos, yPos) / glm::vec2(ptr->size());
    _currPos = 2.f * _currPos - 1.f;
  }

  void Trackball::mouseButtonInput(int button, int action)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      _isTracking = true;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
      _isTracking = false;
    }
  }

  void Trackball::mouseScrollInput(double xScroll, double yScroll)
  {
    _s -= 0.1f * yScroll;
    _s = glm::clamp(_s, 0.1f, 4.0f);
  }
} 