#include "hdi/debug/utils/input.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <algorithm>
#include <iostream>

static void keyboardCallback(GLFWwindow* window, int key, int scan_code, int action, int mods)
{
  // ImGui captures keyboard input, do not forward
  if (ImGui::GetIO().WantCaptureKeyboard) {
    return;
  }

  hdi::dbg::InputManager *ptr = (hdi::dbg::InputManager *) glfwGetWindowUserPointer(window);
  ptr->forwardKeyboardInput(key, action);
}

static void mousePosCallback(GLFWwindow* window, double xPos, double yPos)
{
  // Do forward despite possible ImGui capture
  hdi::dbg::InputManager *ptr = (hdi::dbg::InputManager *) glfwGetWindowUserPointer(window);
  ptr->forwardMousePosInput(xPos, yPos);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  // ImGui captures mouse input, do not forward
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }

  hdi::dbg::InputManager *ptr = (hdi::dbg::InputManager *) glfwGetWindowUserPointer(window);
  ptr->forwardMouseButtonInput(button, action);
}

static void mouseScrollCallback(GLFWwindow* window, double x_scroll, double y_scroll)
{
  // ImGui captures mouse input, do not forward
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }

  hdi::dbg::InputManager *ptr = (hdi::dbg::InputManager *) glfwGetWindowUserPointer(window);
  ptr->forwardMouseScrollInput(x_scroll, y_scroll);
}

namespace hdi::dbg {
  InputManager * InputManager::_currentManager = nullptr;

  InputComponent::InputComponent(bool isInit)
  : _isInit(isInit)
  {
    if (_isInit) {
      init();
    }
  }

  InputComponent::~InputComponent()
  {
    if (_isInit) {
      destr();
    }
  }
  
  void InputComponent::init()
  {
    auto *ptr = InputManager::currentManager();
    if (ptr) {
      ptr->addInputComponent(this);
      _isInit = true;
    }
  }

  void InputComponent::destr()
  {
    auto *ptr = InputManager::currentManager();
    if (_isInit && ptr) {
      ptr->removeInputComponent(this);
    }
    _isInit = false;
  }
  
  void InputComponent::keyboardInput(int key, int action)
  {
    // ...
  }

  void InputComponent::mousePosInput(double xPos, double yPos)
  {
    // ...
  }

  void InputComponent::mouseButtonInput(int button, int action)
  {
    // ...
  }

  void InputComponent::mouseScrollInput(double xScroll, double yScroll)
  {
    // ...
  }

  InputManager::InputManager(const Window &window)
  {
    // Register input callbacks
    GLFWwindow *handle = (GLFWwindow *) window.handle();
    glfwSetWindowUserPointer(handle, this);
    glfwSetKeyCallback(handle, keyboardCallback);
    glfwSetCursorPosCallback(handle, mousePosCallback);
    glfwSetMouseButtonCallback(handle, mouseButtonCallback);
    glfwSetScrollCallback(handle, mouseScrollCallback);

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    const char *version = "#version 450";
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *) window.handle(), true);
    ImGui_ImplOpenGL3_Init(version);

    _currentManager = this;
  }

  InputManager::~InputManager()
  {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    _currentManager = nullptr;
  }

  void InputManager::processInputs() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }

  void InputManager::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

  void InputManager::addInputComponent(InputComponent *ptr)
  {
    if (_components.find(ptr) == _components.end()) {
      _components.insert(ptr);
    }
  }

  void InputManager::removeInputComponent(InputComponent *ptr)
  {
    if (_components.find(ptr) != _components.end()) {
      _components.erase(ptr);
    }
  }

  void InputManager::forwardKeyboardInput(int key, int action)
  {
    for (auto &ptr : _components) {
      ptr->keyboardInput(key, action);
    }
  }

  void InputManager::forwardMousePosInput(double xPos, double yPos)
  {
    for (auto &ptr : _components) {
      ptr->mousePosInput(xPos, yPos);
    }
  }

  void InputManager::forwardMouseButtonInput(int button, int action)
  {
    for (auto &ptr : _components) {
      ptr->mouseButtonInput(button, action);
    }
  }

  void InputManager::forwardMouseScrollInput(double xScroll, double yScroll)
  {
    for (auto &ptr : _components) {
      ptr->mouseScrollInput(xScroll, yScroll);
    }
  }

  InputManager * InputManager::currentManager()
  {
    return _currentManager;
  }
}