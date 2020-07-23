/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
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

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <QApplication>
#include <QCommandLineParser>
#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"

int main(int argc, char *argv[]) {
  try {
    // Prepare parsing of CLI
    QApplication app(argc, argv);
    QCommandLineParser parser;
    parser.setApplicationDescription("Truncate points/dimensions from binary input data");
    parser.addHelpOption();
    parser.addPositionalArgument("input", "Name of input file");
    parser.addPositionalArgument("input_points", "Nr. of input data points");
    parser.addPositionalArgument("input_dimensions", "Nr. of input dimensions");
    parser.addPositionalArgument("output", "Name of output file");
    parser.addPositionalArgument("output_points", "Nr. of output data points");
    parser.addPositionalArgument("output_dimensions", "Nr. of output dimensions");
    parser.process(app);

    // Parse arguments
    const QStringList args = parser.positionalArguments();
    if (args.size() != 6) {
      throw std::runtime_error("Wrong nr. of arguments, should be 6");
    }
    const std::string inputFileName = args.at(0).toStdString();
    const unsigned inputNumPoints = args.at(1).toUInt();
    const unsigned inputNumDimensions = args.at(2).toUInt();
    const std::string outputFileName = args.at(3).toStdString();
    const unsigned outputNumPoints = args.at(4).toUInt();
    const unsigned outputNumDimensions = args.at(5).toUInt();

    // Check input/output size discrepancies
    if (inputNumPoints == 0 || 
        outputNumPoints == 0 || 
        inputNumDimensions == 0 || 
        outputNumDimensions == 0) {
      throw std::logic_error("Number of input/output points/dimensions was 0");
    }
    if (outputNumPoints > inputNumPoints || 
        outputNumDimensions > inputNumDimensions) {
      throw std::logic_error("Number of output points/dimensions larger than input");
    }

    // Open input file
    std::ifstream inputFile(inputFileName, std::ios::in | std::ios::binary);
    if (!inputFile.is_open()) {
      throw std::runtime_error("Could not open input file");
    }
    
    // Open output file
    std::ofstream outputFile(outputFileName, std::ios::out | std::ios::binary);
    if (!outputFile.is_open()) {
      throw std::runtime_error("Could not create output file");
    }

    // Copy specified number of points/dimensions from input to output file
    std::vector<float> buffer(outputNumDimensions);
    for (size_t i = 0; i < outputNumPoints; i++) {
      // Copy over relevant data using buffer, probably not very efficient
      inputFile.read((char *) buffer.data(), sizeof(float) * outputNumDimensions);
      inputFile.seekg(sizeof(float) * (inputNumDimensions - outputNumDimensions), std::ios::cur);
      outputFile.write((char *) buffer.data(), sizeof(float) * outputNumDimensions);
    }

    std::cout << "Output file (" 
              << outputNumPoints << "x" << outputNumDimensions 
              << ") written to " << outputFileName
              << " successfully" << std::endl;
              
    return EXIT_SUCCESS;
  } catch (const std::runtime_error& err) {
    std::cerr << "Runtime err: " << err.what() << std::endl;
  } catch (const std::logic_error& err) {
    std::cerr << "Logic err: " << err.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown err occurred" << std::endl;
  }
  return EXIT_FAILURE;
}