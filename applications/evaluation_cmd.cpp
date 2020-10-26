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
#include "hdi/dimensionality_reduction/gpgpu_sne/utils/types.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_hd_compute.h"
#include "hdi/dimensionality_reduction/gpgpu_sne/gpgpu_kld_compute.h"
#include "hdi/dimensionality_reduction/evaluation.h"

using uint = unsigned;

// Required CLI parameters
std::string hInputFileName;
std::string lInputFileName;
hdi::dr::TsneParameters params;

// Optional CLI parameters with default values
bool doKlDivergence = false;
bool doNNPreservation = false;
bool doLabels = false;

// Timer values
float dataLoadingTime = 0.f;
float simComputationTime = 0.f;
float kldComputationTime = 0.f;
float nnpComputationTime = 0.f;

constexpr char *programTitle = "Evaluation";

void parseCli(hdi::utils::AbstractLog *logger, int argc, char* argv[]) {
  // Configure command line options
  cxxopts::Options options(programTitle, "Compute KL-divergence or NNP of low-dimensional embedding's distribution, given high-dimensional input data");
  options.add_options()
    ("hinput", "Input high-dimensional binary data file (required)", cxxopts::value<std::string>())
    ("linput", "Input low-dimensional binary data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("hdims", "number of high-dimensional input dims (required)", cxxopts::value<unsigned>())
    ("ldims", "number of low-dimensional input dims (required)", cxxopts::value<unsigned>())
    ("p,perplexity", "perplexity parameter of algorithm", cxxopts::value<float>())
    ("h,help", "Print help and exit")
    ("kld", "Compute KL-Divergence", cxxopts::value<bool>())
    ("nnp", "Compute nearest-neighbourhood preservation", cxxopts::value<bool>())
    ("lbl", "Input data file contains labels", cxxopts::value<bool>())
    ;
  options.parse_positional({"hinput", "linput", "size", "hdims", "ldims"});
  options.positional_help("<hinput> <linput> <size> <hdims> <ldims>");
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    hdi::utils::secureLog(logger, options.help());
    std::exit(0);
  }

  // Parse required arguments
  if (!result.count("hinput") || !result.count("linput") ||!result.count("size") 
   || !result.count("hdims") || !result.count("ldims")) {
    hdi::utils::secureLog(logger, options.help());
    std::exit(0);
  }
  hInputFileName = result["hinput"].as<std::string>();
  lInputFileName = result["linput"].as<std::string>();
  params.n = result["size"].as<unsigned>();
  params.nHighDimensions = result["hdims"].as<unsigned>();
  params.nLowDimensions = result["ldims"].as<unsigned>();

  // Parse optional arguments
  if (result.count("perplexity")) {
    params.perplexity = result["perplexity"].as<float>();
  }
  if (result.count("kld")) {
    doKlDivergence = result["kld"].as<bool>();
  }
  if (result.count("nnp")) {
    doNNPreservation = result["nnp"].as<bool>();
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

    if (!doKlDivergence && !doNNPreservation) {
      hdi::utils::secureLog(&logger, "No evaluation metric selected. Use [-h] for help information.");
      std::exit(0);
    }

    // Load input data and labels
    std::vector<float> data;
    std::vector<float> embedding;
    std::vector<uint> labels;
    {
      hdi::utils::secureLog(&logger, "Loading data...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(dataLoadingTime);
      hdi::utils::readBinaryFile(
        hInputFileName,
        data,
        labels,
        params.n,
        params.nHighDimensions,
        doLabels
      );
      hdi::utils::readBinaryFile(
        lInputFileName,
        embedding,
        labels,
        params.n,
        params.nLowDimensions,
        doLabels
      );
    }

    // Create an opengl context
    hdi::dbg::Window window = hdi::dbg::Window::Offscreen();
    window.makeCurrent();

    // Compute subcomponents
    hdi::dr::GpgpuHdCompute hdCompute;
    hdi::dr::GpgpuKldCompute<2> kld2dCompute;
    hdi::dr::GpgpuKldCompute<3> kld3dCompute;

    // Construct high-dimensional joint similarity distribution
    if (doKlDivergence) {
      hdi::utils::secureLog(&logger, "Computing joint similarities...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(simComputationTime);
      hdCompute.init(params);
      hdCompute.compute(data);
    }

    // Move low-dimensional data to gpu
    GLuint positionsBuffer;
    glCreateBuffers(1, &positionsBuffer);
    if (params.nLowDimensions == 2) {
      // Copy over embedding data to a workable type
      std::vector<glm::vec2> buffer(params.n);
      std::memcpy(buffer.data(), embedding.data(), embedding.size() * sizeof(float));

      // Specify buffer
      glNamedBufferStorage(positionsBuffer, buffer.size() * sizeof(glm::vec2), buffer.data(), 0);
    } else if (params.nLowDimensions == 3) {
      using vec = hdi::dr::AlignedVec<3, float>;

      // Copy over unpadded embedding data to a workable type
      std::vector<glm::vec3> _buffer(params.n);
      std::memcpy(_buffer.data(), embedding.data(), embedding.size() * sizeof(float));

      // Make padded to 4x4 bytes
      std::vector<vec> buffer(_buffer.begin(), _buffer.end());

      // Specify buffer
      glNamedBufferStorage(positionsBuffer, buffer.size() * sizeof(vec), buffer.data(), 0);
    }

    // Compute KL-divergence
    if (doKlDivergence) {
      hdi::utils::secureLog(&logger, "\nComputing KL-divergence...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(kldComputationTime);
      float kld;
      if (params.nLowDimensions == 2) {
        kld = kld2dCompute.compute({positionsBuffer}, hdCompute.buffers(), params);
      } else if (params.nLowDimensions == 3) {
        kld = kld3dCompute.compute({positionsBuffer}, hdCompute.buffers(), params);
      }
      hdi::utils::secureLogValue(&logger, "  KL Divergence", kld);
    }

    // Compute mearest neighbour preservation if requested
    if (doNNPreservation) {
      hdi::utils::secureLog(&logger, "\nComputing NNP...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(nnpComputationTime);

      // Prepare data storage
      std::vector<float> precision;
      std::vector<float> recall;
      std::vector<unsigned> points(params.n);
      std::iota(points.begin(), points.end(), 0);

      // Compute p/r... which is an extremely stupid computation that takes basically forever
      hdi::dr::computePrecisionRecall(data, embedding, params, points, precision, recall, 30);

      // Just dump in cout for now
      for (int i = 0; i < precision.size(); i++) {
        std::cout << precision[i] << ", " << recall[i] << '\n';
      }
    }

    // Output computation timings
    hdi::utils::secureLog(&logger, "\nTimings");
    hdi::utils::secureLogValue(&logger, "  Data loading (s)", dataLoadingTime);
    if (doKlDivergence) {
      hdi::utils::secureLogValue(&logger, "  Similarities (s)", simComputationTime);
      hdi::utils::secureLogValue(&logger, "  KLD computation (s)", kldComputationTime);
    }
    if (doNNPreservation) {
      hdi::utils::secureLogValue(&logger, "  NNP computation (s)", nnpComputationTime);
    }

    // Clean up
    glDeleteBuffers(1, &positionsBuffer);
    hdCompute.destr();
  } catch (std::exception& e) { 
    std::cerr << "gpgpu_tsne_cmd caught generic exception: " << e.what() << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}