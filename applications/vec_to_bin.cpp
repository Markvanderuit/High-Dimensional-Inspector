#include <cxxopts/cxxopts.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using uint = unsigned;

std::string dataFilename;
std::string labelsFilename;
std::string outputFilename;
uint n;
uint d;

void parseCli(int argc, char** argv) {
  // Configure command line options
  cxxopts::Options options("fvec/ivec to bin", "Convert AnchorSNE fvec/ivec format to simple binary format");
  options.add_options()
    ("data", "Input fvec data file (required)", cxxopts::value<std::string>())
    ("labels", "Input fvec labels file (required)", cxxopts::value<std::string>())
    ("output", "Output binary data file (required)", cxxopts::value<std::string>())
    ("size", "number of data points (required)", cxxopts::value<unsigned>())
    ("dims", "number of data dims (required)", cxxopts::value<unsigned>())
    ;
  options.parse_positional({"data", "labels", "output", "size", "dims"});
  options.positional_help("<data> <labels> <output> <size> <dims>");
  auto result = options.parse(argc, argv);

  // Help message
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Parse required arguments
  if (!result.count("data") || !result.count("labels") || !result.count("output")
      || !result.count("size") || !result.count("dims")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }
  
  dataFilename = result["data"].as<std::string>();
  labelsFilename = result["labels"].as<std::string>();
  outputFilename = result["output"].as<std::string>();
  n = result["size"].as<uint>();
  d = result["dims"].as<uint>();
}

int main (int argc, char** argv) {
  // Parse command line arguments
  parseCli(argc, argv);

  // Allocate space for storing read data
  std::vector<float> data(n * d);
  std::vector<uint> labels(n);

  // Open input files
  std::ifstream dataIfs(dataFilename, std::ios::in | std::ios::binary);
  if (!dataIfs) {
    std::cerr << "Could not open data file" << std::endl;
    std::exit(0);
  }

  std::ifstream labelsIfs(labelsFilename, std::ios::in | std::ios::binary);
  if (!labelsIfs) {
    std::cerr << "Could not open labels file" << std::endl;
    std::exit(0);
  }

  // Read in data
  // 4 bytes of dump space (yah yah I know)
  uint dump;
  
  // Read labels file
  labelsIfs.read((char *) &dump, sizeof(uint)); // 4by
  labelsIfs.read((char *) labels.data(), n * sizeof(uint));

  // Read data file
  for (uint i = 0; i < n; ++i) {
    dataIfs.read((char *) &dump, sizeof(uint)); // 4by
    dataIfs.read((char *) &data[i * d], d * sizeof(float));
  }

  labelsIfs.close();
  dataIfs.close();

  std::ofstream ofs(outputFilename, std::ios::out | std::ios::binary);
  if (!ofs) {
    std::cerr << "Could not access output file" << std::endl;
    std::exit(0);
  }

  // Write to output file stream
  for (uint i = 0; i < n; ++i) {
    ofs.write((char *) &labels[i], sizeof(uint));
    ofs.write((char *) &data[i * d], d * sizeof(float));
  }

  ofs.close();

  return EXIT_SUCCESS;
}