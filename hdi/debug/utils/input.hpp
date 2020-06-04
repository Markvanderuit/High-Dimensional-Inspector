#pragma once

#include <set>
#include "hdi/debug/utils/window.hpp"

namespace hdi::dbg {
  class InputComponent {
  public:
    InputComponent(bool isInit = false);
    ~InputComponent();

    // Lifetime management
    virtual void init();
    virtual void destr();

    // Input forwarding
    virtual void keyboardInput(int key, int action);
    virtual void mousePosInput(double xPos, double yPos);
    virtual void mouseButtonInput(int button, int action);
    virtual void mouseScrollInput(double xScroll, double yScroll);

  private:
    bool _isInit;
  };

  class InputManager {
  public:
    InputManager(const Window &window);
    ~InputManager();

    // Copy constr/operator is explicitly deleted
    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;

    // Move constr/operator moves resource handle
    InputManager(InputManager&&) noexcept;
    InputManager& operator=(InputManager&&) noexcept;

    // Swap internals with another object
    void swap(InputManager& other) noexcept;

    // Updates all components
    void processInputs();
    void render();

    // Register input components
    void addInputComponent(InputComponent *ptr);
    void removeInputComponent(InputComponent *ptr);

    // Forward input to registered components
    void forwardKeyboardInput(int key, int action);
    void forwardMousePosInput(double xPos, double yPos);
    void forwardMouseButtonInput(int button, int action);
    void forwardMouseScrollInput(double xScroll, double yScroll);

    // Static access
    static InputManager *currentManager();

  private:
    std::set<InputComponent *> _components;

    static InputManager *_currentManager;
  };
}