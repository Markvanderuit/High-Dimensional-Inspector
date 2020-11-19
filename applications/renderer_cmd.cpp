/*
* Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
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

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <glm/glm.hpp>
#include <cxxopts/cxxopts.hpp>
#include "hdi/utils/io.h"
#include "hdi/utils/cout_log.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/debug/utils/window.hpp"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/debug/renderer/embedding.hpp"
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"

using uint = unsigned;

// Required CLI parameters
std::string inputFileName;
uint n;
uint d;

// Optional CLI parameters with default values
bool doLabels = false;

// Timer values
float dataLoadingTime = 0.f;
float boundsComputationTime = 0.f;

// Visual debugger default window settings
constexpr uint windowFlags = 
                          //  hdi::dbg::WindowInfo::bDecorated | 
                           hdi::dbg::WindowInfo::bSRGB |
                           hdi::dbg::WindowInfo::bFocused |
                           hdi::dbg::WindowInfo::bResizable;
constexpr uint windowWidth = 2048u;
constexpr uint windowHeight = 2048u;
constexpr const char *windowTitle = "Renderer";

void parseCli(hdi::utils::AbstractLog *logger, int argc, char* argv[]) {
  // Configure command line options
  cxxopts::Options options(windowTitle, "Visualize low-dimensional embedding with optional labels");
  options.add_options()
    ("input", "Input binary data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of input dims (required)", cxxopts::value<unsigned>())
    ("lbl", "Input data file contains labels", cxxopts::value<bool>())
    ("h,help", "Print help and exit")
    ;
  options.parse_positional({"input", "size", "dims"});
  options.positional_help("<input> <size> <dims>");
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    hdi::utils::secureLog(logger, options.help());
    std::exit(0);
  }

  // Parse required arguments
  if (!result.count("input") || !result.count("size") || !result.count("dims")) {
    hdi::utils::secureLog(logger, options.help());
    std::exit(0);
  }
  inputFileName = result["input"].as<std::string>();
  n = result["size"].as<unsigned>();
  d = result["dims"].as<unsigned>();

  // Parse optional arguments
  if (result.count("lbl")) {
    doLabels = result["lbl"].as<bool>();
  }
}

int main(int argc, char *argv[]) {
  try {
    // Logger which is passed through the program
    hdi::utils::CoutLog logger;

    // Pass input arguments and parse them, setting parameters
    parseCli(&logger, argc, argv);

    // Load input data and labels
    std::vector<float> embedding;
    std::vector<uint> labels;
    {
      hdi::utils::secureLog(&logger, "Loading data...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(dataLoadingTime);
      hdi::utils::readBinaryFile(
        inputFileName,
        embedding,
        labels,
        n,
        d,
        doLabels
      );
    }

    // Create an opengl context (and visible window, if we want one for the visual debugger)
    hdi::dbg::Window window = hdi::dbg::Window({
      windowFlags, 
      windowWidth, 
      windowHeight, 
      windowTitle
    });
    window.makeCurrent();
    window.enableVsync(true);
    window.display();

    // Inputmanager and rendermanager handle input/rendering of visual debugger components
    auto &renderManager = hdi::dbg::RenderManager::instance();
    renderManager.init(window, d, labels);

    // One for each possibility
    hdi::dbg::EmbeddingRenderer<2> embedding2dRenderer;
    hdi::dbg::EmbeddingRenderer<3> embedding3dRenderer;

    // Move embedding to gpu
    GLuint positionsBuffer;
    GLuint boundsBuffer;
    glCreateBuffers(1, &positionsBuffer);
    glCreateBuffers(1, &boundsBuffer);
    std::cout << d << std::endl;
    {
      hdi::utils::secureLog(&logger, "Computing bounds..");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(boundsComputationTime);
      if (d == 2) {
        // Copy over embedding data to a workable type
        std::vector<glm::vec2> buffer(n);
        std::memcpy(buffer.data(), embedding.data(), embedding.size() * sizeof(float));

        // Compute bounds
        std::vector<glm::vec2> bounds(4);
        bounds[0] = std::accumulate(
          buffer.begin(), buffer.end(), glm::vec2(std::numeric_limits<float>::max()),
          [](const auto &a, const auto &b) { return glm::min(a, b); });
        bounds[1] = std::accumulate(
          buffer.begin(), buffer.end(), glm::vec2(std::numeric_limits<float>::min()),
          [](const auto &a, const auto &b) { return glm::max(a, b); });
        bounds[2] = bounds[1] - bounds[0];
        bounds[3] = glm::vec2(1.f) / bounds[2];
        
        // Specify buffers
        glNamedBufferStorage(positionsBuffer, buffer.size() * sizeof(glm::vec2), buffer.data(), 0);
        glNamedBufferStorage(boundsBuffer, 4 * sizeof(glm::vec2), bounds.data(), 0);

        // Pass to renderer
        embedding2dRenderer.init(n, positionsBuffer, boundsBuffer);
      } else if (d == 3) {
        using vec = hdi::dr::AlignedVec<3, float>;

        // Copy over unpadded embedding data to a workable type
        std::vector<glm::vec3> _buffer(n);
        std::memcpy(_buffer.data(), embedding.data(), embedding.size() * sizeof(float));

        // Compute bounds
        std::vector<glm::vec3> _bounds(4);
        _bounds[0] = std::accumulate(
          _buffer.begin(), _buffer.end(), glm::vec3(std::numeric_limits<float>::max()),
          [](const auto &a, const auto &b) {  return glm::min(a, b); });
        _bounds[1] = std::accumulate(
          _buffer.begin(), _buffer.end(), glm::vec3(std::numeric_limits<float>::min()),
          [](const auto &a, const auto &b) { return glm::max(a, b); });
        _bounds[2] = _bounds[1] - _bounds[0];
        _bounds[3] = glm::vec3(1.f) / _bounds[2];

        // Make padded to 4x4 bytes
        std::vector<vec> buffer(_buffer.begin(), _buffer.end());
        std::vector<vec> bounds(_bounds.begin(), _bounds.end());

        // Specify buffers
        glNamedBufferStorage(positionsBuffer, buffer.size() * sizeof(vec), buffer.data(), 0);
        glNamedBufferStorage(boundsBuffer, bounds.size() * sizeof(vec), bounds.data(), 0);

        // Pass to renderer
        embedding3dRenderer.init(n, positionsBuffer, boundsBuffer);
      }
    }

    // Output computation timings
    hdi::utils::secureLog(&logger, "Timings");
    hdi::utils::secureLogValue(&logger, "  Bounds compute (s)", boundsComputationTime);
    hdi::utils::secureLogValue(&logger, "  Data loading (s)", dataLoadingTime);

    // Spin render loop until window is closed
    while (window.canDisplay()) {
      window.processEvents();
      renderManager.render();
      window.display();
    }

    // Clean up
    embedding2dRenderer.destr();
    // embedding3dRenderer.destr();
    renderManager.destr();
    glDeleteBuffers(1, &positionsBuffer);
    glDeleteBuffers(1, &boundsBuffer);
  } catch (std::exception& e) { 
    std::cerr << "gpgpu_tsne_cmd caught generic exception: " << e.what() << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}