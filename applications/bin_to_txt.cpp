#include <cxxopts/cxxopts.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "hdi/utils/io.h"

using uint = unsigned;

std::string dataFilename;
std::string outputFilename;
bool doLabels = false;
uint n;
uint d;

void parseCli(int argc, char** argv) {
  // Configure command line options
  cxxopts::Options options("bin to txt", "Convert simple binary format to txt file");
  options.add_options()
    ("data", "Input binary data file (required)", cxxopts::value<std::string>())
    ("output", "Output txt data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of data dims (required)", cxxopts::value<unsigned>())
    ("lbl", "number of data dims (required)", cxxopts::value<bool>())
    ("h,help", "Print help and exit")
    ;
  options.parse_positional({"data", "output", "size", "dims"});
  options.positional_help("<data> <output> <size> <dims>");
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Parse required arguments
  if (!result.count("data") || !result.count("output")
      || !result.count("size") || !result.count("dims")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }
  dataFilename = result["data"].as<std::string>();
  outputFilename = result["output"].as<std::string>();
  n = result["size"].as<uint>();
  d = result["dims"].as<uint>();

  // Parse optional arguments
  if (result.count("lbl")) {
    doLabels = result["lbl"].as<bool>();
  }
}

int main (int argc, char** argv) {
  // Parse command line arguments
  parseCli(argc, argv);

  // Allocate space for storing read data
  std::vector<float> data(n * d);
  std::vector<uint> labels(n);

  // Read binary file
  std::cout << "Reading input file" << std::endl;
  hdi::utils::readBinaryFile(dataFilename, data, labels, n, d, doLabels);

  std::ofstream ofs(outputFilename, std::ios::out);
  if (!ofs) {
    std::cerr << "Could not access output file" << std::endl;
    std::exit(0);
  }

  // Write to output file stream
  std::cout << "Writing to output file" << std::endl;
  for (uint i = 0; i < n; ++i) {
    if (doLabels) {
      ofs << std::to_string(labels[i]) << ' ';
    }
    for (uint j = 0; j < d; ++j) {
      ofs << std::to_string(data[i * d + j]) << ' ';
    }
    ofs << '\n';
  }
  ofs.close();

  return EXIT_SUCCESS;
}