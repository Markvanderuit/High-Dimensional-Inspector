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

#include <cstdlib>
#include <numeric>
#include <string>
#include <cxxopts/cxxopts.hpp>
#include "hdi/utils/io.h"
#include "hdi/utils/cout_log.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/debug/utils/window.hpp"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/dimensionality_reduction/gpgpu_tsne.h"
#include "hdi/dimensionality_reduction/evaluation.h"

using uint = unsigned;

// Required CLI parameters
std::string inputFileName;
std::string outputFileName;
hdi::dr::TsneParameters params;

// Optional CLI parameters with default values
bool doKldComputation = false;
bool doNnpComputation = false;
bool doVisualisation = false;
bool doLabels = false;

// Timer values
float dataLoadingTime = 0.f;
float simComputationTime = 0.f;
float minComputationTime = 0.f;
float kldComputationTime = 0.f;
float nnpComputationTime = 0.f;
float dataSavingTime = 0.f;

// Visual debugger default window settings
constexpr uint windowFlags = hdi::dbg::WindowInfo::bDecorated
                           | hdi::dbg::WindowInfo::bSRGB 
                           | hdi::dbg::WindowInfo::bFocused
                           | hdi::dbg::WindowInfo::bResizable;
constexpr uint windowWidth = 2560u;
constexpr uint windowHeight = 1440u;
constexpr const char *windowTitle = "tSNE";

void parseCli(hdi::utils::AbstractLog *logger, int argc, char* argv[]) {
  // Configure command line options
  cxxopts::Options options(windowTitle, "Compute low-dimensional embedding from given high-dimensional data");
  options.add_options()
    ("input", "Input binary data file (required)", cxxopts::value<std::string>())
    ("output", "Output binary data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of input dims (required)", cxxopts::value<unsigned>())
    ("d,dimensions", "number of output dims", cxxopts::value<int>())
    ("p,perplexity", "perplexity parameter of algorithm", cxxopts::value<float>())
    ("i,iterations", "number of T-SNE iterations", cxxopts::value<int>())
    ("t,theta", "Single-hierarchy BH-approximation angle", cxxopts::value<float>())
    ("a,theta2", "Dual-hierarchy BH-approximation angle", cxxopts::value<float>())
    ("h,help", "Print help and exit")
    ("lbl", "Input data file contains labels", cxxopts::value<bool>())
    ("vis", "Run the OpenGL visualization", cxxopts::value<bool>())
    ("kld", "Compute KL-Divergence", cxxopts::value<bool>())
    ("nnp", "Compute nearest-neighbourhood preservation", cxxopts::value<bool>())
    ;
  options.parse_positional({"input", "output", "size", "dims"});
  options.positional_help("<input> <output> <size> <dims>");
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    hdi::utils::secureLog(logger, options.help());
    exit(0);
  }

  // Parse required arguments
  if (!result.count("input") || !result.count("output") 
      || !result.count("size") || !result.count("dims")) {
    hdi::utils::secureLog(logger, options.help());
    exit(0);
  }
  inputFileName = result["input"].as<std::string>();
  outputFileName = result["output"].as<std::string>();
  
  params.n = result["size"].as<unsigned>();
  params.nHighDimensions = result["dims"].as<unsigned>();

  // Parse optional arguments
  if (result.count("perplexity")) {
    params.perplexity = result["perplexity"].as<float>();
  }
  if (result.count("iterations")) {
    params.iterations = result["iterations"].as<int>();
  }
  if (result.count("dimensions")) {
    params.nLowDimensions = result["dimensions"].as<int>();
  }
  if (result.count("theta")) {
    params.singleHierarchyTheta = result["theta"].as<float>();
  }
  if (result.count("theta2")) {
    params.dualHierarchyTheta = result["theta2"].as<float>();
  }
  if (result.count("kld")) {
    doKldComputation = result["kld"].as<bool>();
  }
  if (result.count("nnp")) {
    doNnpComputation = result["nnp"].as<bool>();
  }
  if (result.count("vis")) {
    doVisualisation = result["vis"].as<bool>();
  }
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
    std::vector<float> data;
    std::vector<uint> labels;
    {
      hdi::utils::secureLog(&logger, "Loading data...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(dataLoadingTime);
      hdi::utils::readBinaryFile(
        inputFileName,
        data,
        labels,
        params.n,
        params.nHighDimensions,
        doLabels
      );
    }

    // Create an opengl context (and visible window, if we want one for the visual debugger)
    hdi::dbg::Window window = doVisualisation
                            ? hdi::dbg::Window({windowFlags, windowWidth, windowHeight, windowTitle})
                            : hdi::dbg::Window::Offscreen();
    window.makeCurrent();
    window.enableVsync(false);

    // Inputmanager and rendermanager handle input/rendering of visual debugger components
    hdi::dbg::RenderManager renderManager(window);
    if (doVisualisation) {
      renderManager.init(params.nLowDimensions, labels);
    }

    // Initialize tSNE computation
    // This computes the HD joint similarity distribution and then sets up the minimization
    hdi::dr::GpgpuTSNE tSNE;
    tSNE.setLogger(&logger);
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(simComputationTime);
      tSNE.init(data, params);
    }

    if (doVisualisation) {
      window.display();
      window.enableVsync(false);
    }

    // Perform tSNE minimization
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(minComputationTime);
      for (uint i = 0; i < params.iterations; ++i) {
        tSNE.iterate();
        
        // Process debug render components
        if (doVisualisation) {
          window.processEvents();
          renderManager.render();
          window.display();
        }
      }
    }
    
    // Compute KL-divergence if requested
    if (doKldComputation) {
      hdi::utils::secureLog(&logger, "\nComputing KL-Divergence...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(kldComputationTime);
      hdi::utils::secureLogValue(&logger, "  KL Divergence", tSNE.getKLDivergence());
    } 
    
    // Compute mearest neighbour preservation if requested
    if (doNnpComputation) {
      hdi::utils::secureLog(&logger, "\nComputing NNP...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(nnpComputationTime);

      // Prepare data storage
      std::vector<float> precision;
      std::vector<float> recall;
      std::vector<unsigned> points(params.n);
      std::iota(points.begin(), points.end(), 0);

      // Compute p/r... which is an extremely stupid computation that takes basically forever
      hdi::dr::computePrecisionRecall(data, tSNE.getRawEmbedding(), params, points, precision, recall, 30);

      // Just dump in cout for now
      for (int i = 0; i < precision.size(); i++) {
        std::cout << precision[i] << ", " << recall[i] << '\n';
      }
    }

    // Output embedding file
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(dataSavingTime);
      hdi::utils::secureLog(&logger, "\nWriting embedding to file...");
      hdi::utils::writeBinaryFile(
        outputFileName,
        tSNE.getRawEmbedding(),
        labels,
        params.n,
        params.nLowDimensions,
        doLabels
      );
    }

    // Output computation timings
    hdi::utils::secureLog(&logger, "\nTimings");
    hdi::utils::secureLogValue(&logger, "  Data loading (s)", dataLoadingTime);
    hdi::utils::secureLogValue(&logger, "  Similarities (s)", simComputationTime);
    hdi::utils::secureLogValue(&logger, "  Gradient descent (s)", minComputationTime);
    if (doKldComputation) {
      hdi::utils::secureLogValue(&logger, "  KLD computation (s)", kldComputationTime);
    }
    if (doNnpComputation) {
      hdi::utils::secureLogValue(&logger, "  NNP computation (s)", nnpComputationTime);
    }
    hdi::utils::secureLogValue(&logger, "  Data saving (s)", dataSavingTime);

    if (doVisualisation) {
      window.enableVsync(true);

      // Spin render loop until window is closed
      while (window.canDisplay()) {
        window.processEvents();
        renderManager.render();
        window.display();
      }

      renderManager.destr();
    }

    tSNE.destr();
  } catch (std::exception& e) { 
    std::cerr << "gpgpu_tsne_cmd caught generic exception: " << e.what() << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}