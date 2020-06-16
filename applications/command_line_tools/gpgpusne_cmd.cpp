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

#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/dataset_utils.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/debug/utils/window.hpp"
#include "hdi/debug/utils/input.hpp"
#include "hdi/debug/renderer/renderer.hpp"
#include "hdi/data/embedding.h"
#include "hdi/data/panel_data.h"
#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/evaluation.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/kl_divergence_compute.h"
#include <cxxopts/cxxopts.hpp>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

typedef float scalar_type;

// Required CLI parameters
std::string inputFileName;
std::string outputFileName;
unsigned num_data_points;
unsigned num_dimensions;

// Optional CLI parameters with default values
int iterations = 1000;
int exaggeration_iter = 250;
int perplexity = 30;
unsigned embedding_dimensions = 2;
bool doKlDivergence = false;
bool doNNPreservation = false;
bool doVisualisation = false;

// Timer values
float data_loading_time = 0.f;
float similarities_comp_time = 0.f;
float gradient_desc_comp_time = 0.f;
float kl_divergence_comp_time = 0.f;
float nnp_comp_time = 0.f;
float data_saving_time = 0.f;

void loadData(hdi::data::PanelData<scalar_type>& panel_data, std::string filename_data, unsigned int num_points, unsigned int num_dims) {
  std::ifstream file_data(filename_data, std::ios::in | std::ios::binary);
  if (!file_data.is_open()) {
    throw std::runtime_error("data file cannot be found");
  }

  panel_data.clear();
  for (size_t j = 0; j < num_dims; ++j) {
    panel_data.addDimension(std::make_shared<hdi::data::EmptyData>());
  }
  panel_data.initialize();

  for (size_t j = 0; j < num_points; ++j) {
    std::vector<scalar_type> input_data(num_dims);
    // unsigned label = 0u;
    // file_data.read((char *) &label, 1);
    // std::cout << label << std::endl;
    for (size_t i = 0; i < num_dims; ++i) {
      unsigned appo = 0u;
      file_data.read((char *) &appo, 4);
      input_data[i] = scalar_type(appo);
    }
    panel_data.addDataPoint(std::make_shared<hdi::data::EmptyData>(), input_data);
  }
}

void parseCli(int argc, char* argv[]) {
  hdi::utils::CoutLog log;

  // Configure command line options
  cxxopts::Options options("Gpgpu T-SNE", "Compute embedding from given raw input data");
  options.add_options()
    ("input", "Input data file (required)", cxxopts::value<std::string>())
    ("output", "Output data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of dims (required)", cxxopts::value<unsigned>())
    ("d,dimensions", "target embedding dimensions", cxxopts::value<int>())
    ("p,perplexity", "perplexity parameter of algorithm", cxxopts::value<int>())
    ("i,iterations", "number of T-SNE iterations", cxxopts::value<int>())
    ("h,help", "Print help and exit")
    ("vis", "Run the OpenGL visualization", cxxopts::value<bool>())
    ("kld", "Compute KL-Divergence", cxxopts::value<bool>())
    ("nnp", "Compute nearest-neighbourhood preservation", cxxopts::value<bool>())
    ;
  options.parse_positional({"input", "output", "size", "dims"});
  options.positional_help("<input> <output> <size> <dims>");

  // Parse results
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    hdi::utils::secureLog(&log, options.help());
    exit(0);
  }

  if (!result.count("input") || !result.count("output") 
      || !result.count("size") || !result.count("dims")) {
    hdi::utils::secureLog(&log, options.help());
    exit(0);
  }
  inputFileName = result["input"].as<std::string>();
  outputFileName = result["output"].as<std::string>();
  num_data_points = result["size"].as<unsigned>();
  num_dimensions = result["dims"].as<unsigned>();

  // Parse optional arguments
  if (result.count("perplexity")) {
    perplexity = result["perplexity"].as<int>();
  }
  if (result.count("iterations")) {
    iterations = result["iterations"].as<int>();
  }
  if (result.count("dimensions")) {
    embedding_dimensions = result["dimensions"].as<int>();
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
}

int main(int argc, char *argv[]) {
  try {
    parseCli(argc, argv);

    hdi::utils::CoutLog log;
    hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_gen;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;
    hdi::dr::TsneParameters tsne_params;
    hdi::data::Embedding<scalar_type> embedding;
    hdi::data::PanelData<scalar_type> panel_data;

    // Set tsne parameters
    tsne_params._seed = 1;
    tsne_params._perplexity = static_cast<double>(perplexity);
    tsne_params._embedding_dimensionality = embedding_dimensions;
    tsne_params._iterations = iterations;
    tsne_params._mom_switching_iter = exaggeration_iter;
    tsne_params._remove_exaggeration_iter = exaggeration_iter;

    // Load the input data
    {
      hdi::utils::secureLog(&log, "Loading original data...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
      loadData(panel_data, inputFileName, num_data_points, num_dimensions);
    }

    // Compute probability distribution or load from file if it is cached
    std::string cacheFileName = inputFileName 
      + "_cache"
      + "_n=" + std::to_string(num_data_points)
      + "_p=" + std::to_string(perplexity);
    std::ifstream cacheFile(cacheFileName, std::ios::binary);
    if (cacheFile.is_open()) { 
      // There IS a cache file already
      hdi::utils::secureLog(&log, "Loading joint probability distribution from cache file...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
      hdi::data::IO::loadSparseMatrix(distributions, cacheFile);
    } else { 
      // No cache file, start from scratch
      hdi::utils::secureLog(&log, "Computing joint probability distribution...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);

      // Compute joint probability distribution
      prob_gen.setLogger(&log);
      prob_gen_param._perplexity = perplexity;
      prob_gen.computeJointProbabilityDistribution(panel_data.getData().data(),
                                                   panel_data.numDimensions(),
                                                   panel_data.numDataPoints(),
                                                   distributions,
                                                   prob_gen_param);
      
      // Store probability distribution in cache file
      std::ofstream outputFile(cacheFileName, std::ios::binary);
      hdi::data::IO::saveSparseMatrix(distributions, outputFile);
    }

    // Create an opengl context/window
    hdi::dbg::Window window;
    if (doVisualisation) {
      hdi::dbg::WindowInfo windowInfo;
      windowInfo.flags =
                         hdi::dbg::WindowInfo::bDecorated
                       | hdi::dbg::WindowInfo::bSRGB 
                       | hdi::dbg::WindowInfo::bFocused
                       | hdi::dbg::WindowInfo::bResizable;
      windowInfo.width = 1920;
      windowInfo.height = 1920;
      windowInfo.title = "GPGPU T-SNE";
      window = hdi::dbg::Window(windowInfo);
    } else {
      window = hdi::dbg::Window::Offscreen();
    }
    window.enableVsync(false);
    hdi::dbg::InputManager inputManager(window);
    hdi::dbg::RenderManager renderManager(embedding_dimensions);

    // Init T-SNE algorithm and perform minimization
    hdi::dr::GradientDescentTSNE tSNE;
    {
      tSNE.setLogger(&log);
      tSNE.init(distributions, &embedding, tsne_params);
      hdi::utils::secureLog(&log, "Computing gradient descent...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
      if (doVisualisation) {
        // Basic render loop, running T-Sne, processing window input,
        // rendering data, swapping buffers
        window.display();
        for (int i = 0; i < iterations; ++i) {
          tSNE.iterate();
          window.processEvents();
          inputManager.processInputs();
          renderManager.render();
          inputManager.render();
          window.display();
        }
        // Do not destroy tSNE object, needed for further visualization 
      } else {
        // No render loop, just chunk out minimization
        for (int i = 0; i < iterations; ++i) {
          tSNE.iterate();
        }
        tSNE.destr();
      }
    }
    
    // Compute KL-divergence if requested (takes a while in debug mode)
    if (doKlDivergence) {
      hdi::utils::secureLog(&log, "Computing KL-Divergence...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(kl_divergence_comp_time);
      float KL = 0.f;
      if (embedding_dimensions == 2) {
        KL = hdi::dr::KlDivergenceCompute<2>().compute(&embedding, tsne_params, distributions);
      } else if (embedding_dimensions == 3) {
        KL = hdi::dr::KlDivergenceCompute<3>().compute(&embedding, tsne_params, distributions);
      }
      hdi::utils::secureLogValue(&log, "  KL Divergence", KL);
      hdi::utils::secureLog(&log, "");
    } 
    
    // Compute mearest neighbour preservation if requested (takes a long while with these settings)
    if (doNNPreservation) {
      hdi::utils::secureLog(&log, "Computing NNP...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(nnp_comp_time);
      std::vector<scalar_type> precision;
      std::vector<scalar_type> recall;
      unsigned K = 30;
      std::vector<unsigned> points(num_data_points);
      std::iota(points.begin(), points.end(), 0);
      hdi::dr::computePrecisionRecall(panel_data, embedding, points, precision, recall, K);
      for (int i = 0; i < precision.size(); i++) {
        std::cout << precision[i] << ", " << recall[i] << '\n';
      }
    }

    // Output embedding file
    hdi::utils::secureLog(&log, "Writing embedding to file...");
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_saving_time);
      std::ofstream output_file(outputFileName, std::ios::out | std::ios::binary);
      embedding.removePadding();
      output_file.write(reinterpret_cast<const char*>(embedding.getContainer().data()), sizeof(scalar_type)*embedding.getContainer().size());
    }

    hdi::utils::secureLog(&log, "Timings");
    hdi::utils::secureLogValue(&log, "  Data loading (s)", data_loading_time);
    hdi::utils::secureLogValue(&log, "  Similarities (s)", similarities_comp_time);
    hdi::utils::secureLogValue(&log, "  Gradient descent (s)", gradient_desc_comp_time);
    if (doKlDivergence) {
      hdi::utils::secureLogValue(&log, "  KL computation (s)", kl_divergence_comp_time);
    }
    if (doNNPreservation) {
      hdi::utils::secureLogValue(&log, "  NNP computation (s)", nnp_comp_time);
    }
    hdi::utils::secureLogValue(&log, "  Data saving (s)", data_saving_time);

    // Spin render loop until window closed or application killed
    if (doVisualisation) {
      window.enableVsync(true);
      while (window.canDisplay()) {
        window.processEvents();
        inputManager.processInputs();
        renderManager.render();
        inputManager.render();
        window.display();
      }
      tSNE.destr();
    }

  } catch (std::exception& e) { 
    std::cerr << "Caught exception: " << e.what() << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}