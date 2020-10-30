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

#include <iostream> 
#include <fstream>
#include <stdexcept>
#include "hdi/utils/io.h"

namespace hdi::utils {
  void readBinaryFile(const std::string &fileName, 
                      std::vector<float> &data,
                      std::vector<uint> &labels, 
                      uint n,
                      uint d,
                      bool withLabels)
  {
    std::ifstream ifs(fileName, std::ios::in | std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Input file cannot be accessed: " + fileName);
    }

    // Clear vectors and create space to store data in
    data = std::vector<float>(n * d);
    if (withLabels) {
      labels = std::vector<uint>(n);
    }

    // Read data, either in a single call (no labels) or intermittently (to extract labels)
    if (withLabels) {
      for (uint i = 0; i < n; ++i) {
        ifs.read((char *) &labels[i], sizeof(uint));
        ifs.read((char *) &data[d * i], d * sizeof(float));
      }
    } else {
      ifs.read((char *) data.data(), data.size() * sizeof(float));
    }
  }

  void writeBinaryFile(const std::string &fileName,
                       const std::vector<float> &data,
                       const std::vector<uint> &labels,
                       uint n,
                       uint d,
                       bool withLabels)
  {
    std::ofstream ofs(fileName, std::ios::out | std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    if (withLabels) {
      for (uint i = 0; i < n; ++i) {
        ofs.write((char *) &labels[i], sizeof(uint));
        ofs.write((char *) &data[d * i], d * sizeof(float));
      }
    } else {
      ofs.write((char *) data.data(), data.size() * sizeof(float));
    }
  }

  
  void writeTextValuesFile(const std::string &fileName, const std::vector<std::string> &values)
  {
    std::ofstream ofs (fileName, std::ios::out);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    for (const auto &v : values) {
      ofs << v << '\n';
    }
  }
} // namespace hdi::utils