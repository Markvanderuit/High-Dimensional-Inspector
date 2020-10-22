/*
*
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
*
*/

#include <cstdlib>
#include <fstream>
#include <string>
#include <cxxopts/cxxopts.hpp>
#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/debug/utils/window.hpp"
#include "hdi/debug/utils/input.hpp"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/data/panel_data.h"
#include "hdi/data/empty_data.h"
#include "hdi/dimensionality_reduction/gpgpu_tsne.h"
#include "hdi/dimensionality_reduction/evaluation.h"

using uint = unsigned;

// Required CLI parameters
std::string inputFileName;
std::string outputFileName;
uint n;
uint nHighDimensions;

// Optional CLI parameters with default values
uint iterations = 1000;
uint nLowDimensions = 2;
float perplexity = 30.f;
float singleHierarchyTheta = 0.5;
float dualHierarchyTheta = 0.5;
bool doKlDivergence = false;
bool doNNPreservation = false;
bool doVisualisation = false;
bool doLabelExtraction = false;

// Timer values
float data_loading_time = 0.f;
float similarities_comp_time = 0.f;
float gradient_desc_comp_time = 0.f;
float kl_divergence_comp_time = 0.f;
float nnp_comp_time = 0.f;
float data_saving_time = 0.f;

// Visual debugger default window settings
constexpr uint windowFlags = hdi::dbg::WindowInfo::bDecorated
                           | hdi::dbg::WindowInfo::bSRGB 
                           | hdi::dbg::WindowInfo::bFocused
                           | hdi::dbg::WindowInfo::bResizable;
constexpr uint windowWidth = 2560u;
constexpr uint windowHeight = 1440u;
constexpr char *windowTitle = "GPGPU tSNE";
 
void loadData(hdi::data::PanelData<float>& panelData, 
              std::vector<unsigned> &labelData, 
              std::string filename_data, 
              unsigned int num_points, 
              unsigned int num_dims) 
{
  std::ifstream file_data(filename_data, std::ios::in | std::ios::binary);
  if (!file_data.is_open()) {
    throw std::runtime_error("data file cannot be found");
  }
  
  panelData.clear();
  for (size_t j = 0; j < num_dims; ++j) {
    panelData.addDimension(std::make_shared<hdi::data::EmptyData>());
  }
  panelData.initialize();

  if (doLabelExtraction) {
    labelData.resize(num_points);
  }

  std::vector<float> buffer(num_dims);
  for (size_t j = 0; j < num_points; ++j) {
    // Read 4-byte label if label extraction is required
    if (doLabelExtraction) {
      file_data.read((char *) &labelData[j], sizeof(uint));
    }
    // Read 4-byte point data for entire row
    file_data.read((char *) buffer.data(), sizeof(float) * buffer.size());
    panelData.addDataPoint(std::make_shared<hdi::data::EmptyData>(), buffer);
  }
}

void parseCli(hdi::utils::AbstractLog *logger, int argc, char* argv[]) {
  // Configure command line options
  cxxopts::Options options("Gpgpu T-SNE", "Compute low-dimensional embedding from given high-dimensional data");
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
  n = result["size"].as<unsigned>();
  nHighDimensions = result["dims"].as<unsigned>();

  // Parse optional arguments
  if (result.count("perplexity")) {
    perplexity = result["perplexity"].as<float>();
  }
  if (result.count("iterations")) {
    iterations = result["iterations"].as<int>();
  }
  if (result.count("dimensions")) {
    nLowDimensions = result["dimensions"].as<int>();
  }
  if (result.count("theta")) {
    singleHierarchyTheta = result["theta"].as<float>();
  }
  if (result.count("theta2")) {
    dualHierarchyTheta = result["theta2"].as<float>();
  }
  if (result.count("kld")) {
    doKlDivergence = result["kld"].as<bool>();
  }
  if (result.count("nnp")) {
    doNNPreservation = result["nnp"].as<bool>();
  }
  if (result.count("vis")) {
    doVisualisation = result["vis"].as<bool>();
  }
  if (result.count("lbl")) {
    doLabelExtraction = result["lbl"].as<bool>();
  }
}

int main(int argc, char *argv[]) {
  try {
    // Logger which is passed through the program
    hdi::utils::CoutLog logger;

    // Pass input arguments and parse them, setting parameters
    parseCli(&logger, argc, argv);

    // Load input data and labels
    hdi::data::PanelData<float> panelData;
    std::vector<unsigned> labelData;
    {
      hdi::utils::secureLog(&logger, "Loading data...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
      loadData(panelData, labelData, inputFileName, n, nHighDimensions);
    }

    // Create an opengl context (and visible window, if we want one for the visual debugger)
    hdi::dbg::Window window = doVisualisation
                            ? hdi::dbg::Window({windowFlags, windowWidth, windowHeight, windowTitle})
                            : hdi::dbg::Window::Offscreen();
    window.makeCurrent();
    window.enableVsync(false);

    // Inputmanager and rendermanager handle input/rendering of visual debugger components
    hdi::dbg::InputManager inputManager(window);
    hdi::dbg::RenderManager renderManager;
    if (doVisualisation) {
      renderManager.init(nLowDimensions, labelData);
    }

    // Set CLI arguments in parameter object passed through the program
    hdi::dr::TsneParameters params;
    params.seed = 1;
    params.perplexity = perplexity;
    params.n = n;
    params.nHighDimensions = nHighDimensions;
    params.nLowDimensions = nLowDimensions;
    params.iterations = iterations;
    params.singleHierarchyTheta = singleHierarchyTheta;
    params.dualHierarchyTheta = dualHierarchyTheta;

    // Initialize tSNE computation
    // This computes the HD joint similarity distribution and then sets up the minimization
    hdi::dr::GpgpuTSNE tSNE;
    tSNE.setLogger(&logger);
    tSNE.init(panelData, params);

    if (doVisualisation) {
      window.display();
      window.enableVsync(false);
    }

    // Perform tSNE minimization
    for (uint i = 0; i < iterations; ++i) {
      tSNE.iterate();
      
      // Process debug render components
      if (doVisualisation) {
        window.processEvents();
        inputManager.processInputs();
        renderManager.render();
        inputManager.render();
        window.display();
      }
    }
    
    // Compute KL-divergence if requested (takes a while in debug mode)
    if (doKlDivergence) {
      hdi::utils::secureLog(&logger, "Computing KL-Divergence...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(kl_divergence_comp_time);
      hdi::utils::secureLogValue(&logger, "  KL Divergence", tSNE.getKLDivergence());
      hdi::utils::secureLog(&logger, "");
    } 
    
    // Compute mearest neighbour preservation if requested (takes a long while with these settings)
    /* if (doNNPreservation) {
      hdi::utils::secureLog(&logger, "Computing NNP...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(nnp_comp_time);
      std::vector<float> precision;
      std::vector<float> recall;
      unsigned K = 30;
      std::vector<unsigned> points(n);
      std::iota(points.begin(), points.end(), 0);
      hdi::dr::computePrecisionRecall(panelData, embedding, points, precision, recall, K);
      for (int i = 0; i < precision.size(); i++) {
        std::cout << precision[i] << ", " << recall[i] << '\n';
      }
    } */

    // Output embedding file
    hdi::utils::secureLog(&logger, "Writing embedding to file...");
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_saving_time);
      auto embedding = tSNE.getRawEmbedding();
      std::ofstream output_file(outputFileName, std::ios::out | std::ios::binary);
      output_file.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
    }

    // Output computation timings
    hdi::utils::secureLog(&logger, "Timings");
    hdi::utils::secureLogValue(&logger, "  Data loading (s)", data_loading_time);
    hdi::utils::secureLogValue(&logger, "  Similarities (s)", similarities_comp_time);
    hdi::utils::secureLogValue(&logger, "  Gradient descent (s)", gradient_desc_comp_time);
    if (doKlDivergence) {
      hdi::utils::secureLogValue(&logger, "  KL computation (s)", kl_divergence_comp_time);
    }
    if (doNNPreservation) {
      hdi::utils::secureLogValue(&logger, "  NNP computation (s)", nnp_comp_time);
    }
    hdi::utils::secureLogValue(&logger, "  Data saving (s)", data_saving_time);

    if (doVisualisation) {
      window.enableVsync(true);

      // Spin render loop until window is closed
      while (window.canDisplay()) {
        window.processEvents();
        inputManager.processInputs();
        renderManager.render();
        inputManager.render();
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