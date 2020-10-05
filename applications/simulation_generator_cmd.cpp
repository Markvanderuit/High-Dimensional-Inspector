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
#include "hdi/utils/scoped_timers.h"
#include <cxxopts/cxxopts.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <string>

// Required CLI parameters
std::string outputFileName;
unsigned num_data_points;
unsigned num_dimensions;
unsigned num_clusters;

// Optional CLI parameters with default values
bool addLabels = false;

void parseCli(int argc, char* argv[]) {
  hdi::utils::CoutLog log;

  // Configure command line options
  cxxopts::Options options("Simulation generator", "Generate simulated point data with clusters");
  options.add_options()
    ("output", "Output data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of dims (required)", cxxopts::value<unsigned>())
    ("clusters", "number of clusters (required)", cxxopts::value<unsigned>())
    ("h,help", "Print help and exit")
    ("lbl", "Output data file contains labels", cxxopts::value<bool>())
    ;
  options.parse_positional({"output", "size", "dims", "clusters"});
  options.positional_help("<output> <size> <dims> <clusters>");

  // Parse results
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    hdi::utils::secureLog(&log, options.help());
    exit(0);
  }
  
  // Parse required arguments
  if (!result.count("output") || !result.count("size") 
      || !result.count("dims") || !result.count("clusters")) {
    hdi::utils::secureLog(&log, options.help());
    exit(0);
  }
  outputFileName = result["output"].as<std::string>();
  num_data_points = result["size"].as<unsigned>();
  num_dimensions = result["dims"].as<unsigned>();
  num_clusters = result["clusters"].as<unsigned>();

  // Parse optional arguments
  if (result.count("lbl")) {
    addLabels = result["lbl"].as<bool>();
  }
}

template <typename Int> 
inline
Int ceilDiv(Int n, Int div) {
  return (n + div - 1) / div;
}

int main(int argc, char *argv[]) {
  try {
    parseCli(argc, argv);
    hdi::utils::CoutLog log;

    hdi::utils::secureLog(&log, "Generating simulated data");
    hdi::utils::secureLogValue(&log, "  size", num_data_points);
    hdi::utils::secureLogValue(&log, "  dims", num_dimensions);
    hdi::utils::secureLogValue(&log, "  clus", num_clusters);
    
    // Initialize mersenne twister and random distributions
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> normDistr(0.f, .5f);
    std::uniform_real_distribution<float> unifDistr(0.f, 1.f);

    // Some memory to generate data in
    std::vector<unsigned> labelData(num_data_points);
    std::vector<float> pointData(num_data_points * num_dimensions);
    std::vector<float> clusterData(num_clusters * num_dimensions);

    // Generate uniformly distributed cluster location
    for (size_t c = 0; c < num_clusters; c++) {
      for (size_t j = 0; j < num_dimensions; j++) {
        clusterData[c * num_dimensions + j]  = unifDistr(gen);
      }
    }

    // Generate normal distributed points around each cluster location
    for (size_t c = 0; c < num_clusters; c++) {
      // Cluster start and range
      const size_t range = ceilDiv(num_data_points, num_clusters);
      const size_t begin = c * range;
      const size_t end = std::min(begin + range, static_cast<size_t>(num_data_points));

      for (size_t i = begin; i < end; i++) {
        labelData[i] = static_cast<unsigned>(c);
        for (size_t j = 0; j < num_dimensions; j++) {
          pointData[i * num_dimensions + j] = normDistr(gen) + clusterData[c * num_dimensions + j]; // static_cast<float>(c);
        }
      }
    }
    
    // Write to file
    std::ofstream file(outputFileName, std::ios::out | std::ios::binary);
    for (size_t i = 0; i < num_data_points; i++) {
      // Write label
      file.write((char *) &labelData[i], sizeof(unsigned));

      // Write points
      file.write((char *) &pointData[i * num_dimensions], num_dimensions * sizeof(float));
    }

  } catch (std::exception& e) { 
    std::cerr << "Caught exception: " << e.what() << std::endl; 
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}