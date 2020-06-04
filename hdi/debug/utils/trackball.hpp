#pragma once

#include "glm/glm.hpp"
#include "hdi/debug/utils/input.hpp"

namespace hdi::dbg {
  class Trackball : public InputComponent {
  public:
    Trackball();

    void compute();
    glm::mat4 matrix() const;

    // Input handler overrides
    void mousePosInput(double xPos, double yPos) override;
    void mouseButtonInput(int button, int action) override;
    void mouseScrollInput(double xScroll, double yScroll) override;
    
  private:
    glm::vec2 _prevPos, _currPos;
    glm::vec3 _eye, _up;
    glm::mat4 _r, _m;
    float _s;
    bool _isTracking;
  };
}