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

#include "hdi/dimensionality_reduction/tsne.h"
#include "hdi/utils/cout_log.h"
#include "hdi/utils/log_helper_functions.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QIcon>
#include <iostream>
#include <memory>
#include <fstream>
#include <cstdio>
#include "hdi/visualization/pointcloud_canvas_qobj.h"
#include "hdi/visualization/pointcloud_drawer_fixed_color.h"
#include "hdi/visualization/pointcloud_drawer_labels.h"

int main(int argc, char *argv[])
{
  try{
    QApplication app(argc, argv);
    QIcon icon;
    icon.addFile(":/hdi16.png");
    icon.addFile(":/hdi32.png");
    icon.addFile(":/hdi64.png");
    icon.addFile(":/hdi128.png");
    icon.addFile(":/hdi256.png");
    app.setWindowIcon(icon);

    QCoreApplication::setApplicationName("3D Embedding Viewer");
    QCoreApplication::setApplicationVersion("0.1");

    QCommandLineParser parser;
    parser.setApplicationDescription("Point-based visualization of a 3D embedding");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("embedding", QCoreApplication::translate("main", "Embedding."));

    QCommandLineOption labels_option(QStringList() << "l" << "labels",
      QCoreApplication::translate("main", "Provide labels file"),
      QCoreApplication::translate("main", "labels"));
    parser.addOption(labels_option);

    // Process the actual command line arguments given by the user
    parser.process(app);

    const QStringList args = parser.positionalArguments();
    if(args.size() < 1){
      std::cout << "Not enough arguments!" << std::endl;
      return -1;
    }

    std::ifstream input_file (args[0].toStdString(), std::ios::in|std::ios::binary|std::ios::ate);

    const unsigned int num_data_points = input_file.tellg() / sizeof(float)/3;
    std::vector<float> data(num_data_points*3);
    std::vector<uint32_t> labels(num_data_points);

    input_file.seekg (0, std::ios::beg);
    input_file.read (reinterpret_cast<char*>(data.data()), sizeof(float) * data.size() * 3);
    input_file.close();

    // Read labels if they are provided, otherwise all same label
    if (parser.isSet(labels_option)) {
      std::ifstream label_file(parser.value(labels_option).toStdString(), std::ios::in|std::ios::binary|std::ios::ate);
      label_file.seekg (0, std::ios::beg);
      label_file.read (reinterpret_cast<char*>(labels.data()), sizeof(uint32_t) * labels.size());
      label_file.close();
    }

    float min_x(std::numeric_limits<float>::max());
    float max_x(-std::numeric_limits<float>::max());
    float min_y(std::numeric_limits<float>::max());
    float max_y(-std::numeric_limits<float>::max());
    float min_z(std::numeric_limits<float>::max());
    float max_z(-std::numeric_limits<float>::max());

    for(int i = 0; i < num_data_points; ++i){
      min_x = std::min(min_x, data[i * 3]);
      max_x = std::max(max_x, data[i * 3]);
      min_y = std::min(min_y, data[i * 3 + 1]);
      max_y = std::max(max_y, data[i * 3 + 1]);
      min_z = std::min(min_z, data[i * 3 + 2]);
      max_z = std::max(max_z, data[i * 3 + 2]);
    }  

    std::cout 
            << "Boundaries: " << "\n"
            << "x: " << min_x << ", " << max_x << "\n"
            << "y: " << min_y << ", " << max_y << "\n"
            << "z: " << min_z << ", " << max_z << std::endl;

    auto tr = QVector3D(max_x, max_y, max_z);
    auto bl = QVector3D(min_x, min_y, min_z);
    auto offset = (tr - bl) * 0.125;

    hdi::viz::PointcloudCanvas viewer;
    viewer.setBackgroundColors(qRgb(240,240,240),qRgb(200,200,200));
    viewer.setSelectionColor(qRgb(50,50,50));
    viewer.resize(500,500);
    viewer.show();
    viewer.setTopRightCoordinates(QVector2D(tr + offset));
    viewer.setBottomLeftCoordinates(QVector2D(bl - offset));
    viewer.setEmbeddingBounds(bl - offset, tr + offset);

    std::unique_ptr<hdi::viz::AbstractPointcloudDrawer> drawer_ptr;
    if (parser.isSet(labels_option)) {
      // Color palette
      std::map<unsigned int, QColor> palette;
      palette[0] = qRgb(16,78,139);
      palette[1] = qRgb(139,90,43);
      palette[2] = qRgb(138,43,226);
      palette[3] = qRgb(0,128,0);
      palette[4] = qRgb(255,150,0);
      palette[5] = qRgb(204,40,40);
      palette[6] = qRgb(131,139,131);
      palette[7] = qRgb(0,205,0);
      palette[8] = qRgb(20,20,20);
      palette[9] = qRgb(0, 150, 255);

      auto ptr = std::make_unique<hdi::viz::PointcloudDrawerLabels>();
      ptr->initialize(viewer.context());
      ptr->setData(data.data(), labels.data(), palette, num_data_points);
      // ptr->setData(data.data(), flags.data(), num_data_points);
      ptr->setAlpha(0.85);
      ptr->setPointSize(8);
      drawer_ptr = std::move(ptr);
    } else {
      auto ptr = std::make_unique<hdi::viz::PointcloudDrawerFixedColor>();
      ptr->initialize(viewer.context());
      ptr->setData(data.data(), num_data_points);
      ptr->setAlpha(0.85);
      ptr->setPointSize(8);
      drawer_ptr = std::move(ptr);
    }
    viewer.addDrawer(drawer_ptr.get());

    return app.exec();
  }
  catch(std::logic_error& ex){ std::cout << "Logic error: " << ex.what();}
  catch(std::runtime_error& ex){ std::cout << "Runtime error: " << ex.what();}
  catch(...){ std::cout << "An unknown error occurred";}
}
