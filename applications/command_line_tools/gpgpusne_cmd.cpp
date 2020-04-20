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
#include "hdi/utils/visual_utils.h"
#include "hdi/utils/scoped_timers.h"
#include "hdi/data/embedding.h"
#include "hdi/data/panel_data.h"
#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/abstract_gradient_descent_tsne.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_3d.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/evaluation.h"
#include <QApplication>
#include <QCommandLineParser>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>

#define USE_GLFW_CONTEXT
#ifdef USE_GLFW_CONTEXT
#include "glfw/glfw3.h"
#else
#include <QWindow>
#include <QOpenGLContext>
#endif

// #define USE_OLD_2D_TSNE
#ifdef USE_OLD_2D_TSNE
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"
#else
#include "hdi/dimensionality_reduction/gradient_descent_tsne_2d.h"
#endif

typedef float scalar_type;

#define GLSL(name, version, shader) \
  static const char * name = \
  "#version " #version "\n" #shader

#ifdef USE_GLFW_CONTEXT
class UniqueContext {
public:
  UniqueContext() {
    if (!glfwInit()) {
      throw std::runtime_error("Could not initialize GLFW");
    }

    // Employ window hints to create a OpenGL 4.5 core profile context
    // and no visible window
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR , 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR , 5);

    pWindow = glfwCreateWindow(980, 980, "", nullptr, nullptr);
    if (!pWindow) {
      throw std::runtime_error("Could not create window");
    }

    glfwMakeContextCurrent(pWindow);

    if (!gladLoadGL()) {
      throw std::runtime_error("Could not initialize GLAD");
    }

    glfwSwapInterval(0);
  }

  ~UniqueContext() {
    glfwDestroyWindow(pWindow);
    glfwTerminate();
  }

  void swapBuffers() {
    glfwSwapBuffers(pWindow);
  }

  void poll() {
    glfwPollEvents();
  }

  bool shouldClose() {
    return glfwWindowShouldClose(pWindow);
  }

private:
  GLFWwindow* pWindow;
};
#else
class UniqueContext : public QWindow {
public:
  UniqueContext() {
    setSurfaceType(QWindow::OpenGLSurface);
    _context = new QOpenGLContext(this);
    auto fmt = requestedFormat();
    fmt.setMajorVersion(4);
    fmt.setMinorVersion(5);
    fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
    _context->setFormat(fmt);
    if (!_context->create())
      qFatal("Cannot create requested OpenGL context.");
    
    create();

    _context->makeCurrent(this);
    if (!gladLoadGL()) {
      qFatal("No OpenGL context is currently bound, therefore OpenGL function loading has failed.");
    }
  }

  ~UniqueContext() {
    _context->doneCurrent();
  }

private:
  QOpenGLContext* _context;
};
#endif

void loadData(hdi::data::PanelData<scalar_type>& panel_data, std::string filename_data, unsigned int num_points, unsigned int num_dims) {
  panel_data.clear();
  std::ifstream file_data(filename_data, std::ios::in | std::ios::binary);

  if (!file_data.is_open()) {
    throw std::runtime_error("data file cannot be found");
  }

  for (size_t j = 0; j < num_dims; ++j) {
    panel_data.addDimension(std::make_shared<hdi::data::EmptyData>());
  }
  panel_data.initialize();

  {
    for (size_t j = 0; j < num_points; ++j) {
      std::vector<scalar_type> input_data(num_dims);
      for (size_t i = 0; i < num_dims; ++i) {
        float appo;
        file_data.read((char*)&appo, 4);
        input_data[i] = scalar_type(appo);
      }
      panel_data.addDataPoint(std::make_shared<hdi::data::EmptyData>(), input_data);
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    QApplication app(argc, argv); // QT application really only serves for parsing at this point
    QCommandLineParser parser;
    hdi::utils::CoutLog log;

    // Configure parser
    parser.setApplicationDescription("Compute embedding from given raw input data");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("data", QApplication::translate("main", "Name of input file."));
    parser.addPositionalArgument("output", QCoreApplication::translate("main", "Name of output file."));
    parser.addPositionalArgument("num_data_points", QCoreApplication::translate("main", "Num of data-points."));
    parser.addPositionalArgument("num_dimensions", QCoreApplication::translate("main", "Num of dimensions."));

    // Iterations
    QCommandLineOption iterations_option(QStringList() << "i" << "iterations",
        QCoreApplication::translate("main", "Run the gradient for <iterations>."),
        QCoreApplication::translate("main", "iterations"));
    parser.addOption(iterations_option);

    // Target dimensions
    QCommandLineOption target_dimensions_option(QStringList() << "d" << "target_dimensions",
        QCoreApplication::translate("main", "Reduce the dimensionality to <target_dimensions>."),
        QCoreApplication::translate("main", "target_dimensions"));
    parser.addOption(target_dimensions_option);

    // Perplexity
    QCommandLineOption perplexity_option(QStringList() << "p" << "perplexity",
        QCoreApplication::translate("main", "Use perplexity value of <perplexity>."),
        QCoreApplication::translate("main", "perplexity"));
    parser.addOption(perplexity_option);

    // Process the actual command line arguments given by the user
    parser.process(app);

    const QStringList args = parser.positionalArguments();
    if (args.size() != 4) {
      hdi::utils::secureLog(&log, "Not enough arguments!");
      return EXIT_FAILURE;
    }

    int iterations = 1000;
    int exaggeration_iter = 250;
    int perplexity = 30;
    double theta = 0.5;
    unsigned embedding_dimensions = 2;

    // Parse arguments
    std::string inputFileName = args.at(0).toStdString();
    std::string outputFileName = args.at(1).toStdString();
    
    unsigned num_data_points = std::atoi(args.at(2).toStdString().c_str());
    unsigned num_dimensions = std::atoi(args.at(3).toStdString().c_str());

    if(parser.isSet(iterations_option)){
      iterations  = std::atoi(parser.value(iterations_option).toStdString().c_str());
    }
    if(parser.isSet(target_dimensions_option)){
      embedding_dimensions = std::atoi(parser.value(target_dimensions_option).toStdString().c_str());
    }
    if(parser.isSet(perplexity_option)){
      perplexity = std::atoi(parser.value(perplexity_option).toStdString().c_str());
    }

    typedef float scalar_type;

    hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_gen;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_param;
    hdi::dr::TsneParameters tsne_params;
    hdi::data::Embedding<scalar_type> embedding;
    hdi::data::PanelData<scalar_type> panel_data;
    std::vector<uint32_t> labels;

    // Set tsne parameters
    tsne_params._seed = 1;
    tsne_params._perplexity = static_cast<double>(perplexity);
    tsne_params._embedding_dimensionality = embedding_dimensions;
    tsne_params._iterations = iterations;
    tsne_params._mom_switching_iter = exaggeration_iter;
    tsne_params._remove_exaggeration_iter = exaggeration_iter;

    // Timer values
    float data_loading_time = 0.f;
    float similarities_comp_time = 0.f;
    float gradient_desc_comp_time = 0.f;
    float kl_divergence_comp_time = 0.f;
    float data_saving_time = 0.f;

    // Compute probability distribution or load from file if it is cached
    std::string cacheFileName = inputFileName + "_cache_p" + std::to_string(perplexity);
    std::ifstream cacheFile(cacheFileName, std::ios::binary);
    if (cacheFile.is_open()) { // There IS a cache file already
      hdi::utils::secureLog(&log, "Loading probability distribution from cache file...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
      hdi::data::IO::loadSparseMatrix(distributions, cacheFile);
    } else  { // No cache file, start from scratch
      {
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
      }
      // Load the input data
      hdi::utils::secureLog(&log, "Loading original data...");
      loadData(panel_data, inputFileName, num_data_points, num_dimensions);

      // Compute probability distribution
      hdi::utils::secureLog(&log, "Computing probability distribution...");
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
      prob_gen.setLogger(&log);
      prob_gen_param._perplexity = perplexity;
      prob_gen.computeProbabilityDistributions(panel_data.getData().data(), panel_data.numDimensions(), panel_data.numDataPoints(), distributions, prob_gen_param);

      // Store probability distribution in cache file
      std::ofstream outputFile(cacheFileName, std::ios::binary);
      hdi::data::IO::saveSparseMatrix(distributions, outputFile);
    }

    // OpenGL context so we can make OpenGL calls
    UniqueContext context;

    // Compute embedding
    std::unique_ptr<hdi::dr::AbstractGradientDescentTSNE> tSNE;
    if (tsne_params._embedding_dimensionality == 2) {
#ifdef USE_OLD_2D_TSNE
      tSNE = std::make_unique<hdi::dr::GradientDescentTSNETexture>();
#else 
      tSNE = std::make_unique<hdi::dr::GradientDescentTSNE2D>();
#endif 
    } else if (tsne_params._embedding_dimensionality == 3) {
      tSNE = std::make_unique<hdi::dr::GradientDescentTSNE3D>();
    } else {
      throw std::logic_error("Number of embedding dimensions is not supported!");
    }
    tSNE->setLogger(&log);
    tSNE->initialize(distributions, &embedding, tsne_params);
    
    {
      hdi::utils::secureLog(&log, "Computing gradient descent...");  
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
      for (int iter = 0; iter < iterations; ++iter) {
        tSNE->iterate();
        context.swapBuffers();
        context.poll();
      }
    }

    hdi::utils::secureLog(&log, "Removing padding from embedding data...");  
    embedding.removePadding();

    double KL = 0.0;
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(kl_divergence_comp_time);
      KL = tSNE->computeKullbackLeiblerDivergence();
    } 

    //Output
    hdi::utils::secureLog(&log, "Writing embedding to file...");
    {
      hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_saving_time);
      {
        std::ofstream output_file(outputFileName, std::ios::out | std::ios::binary);
        output_file.write(reinterpret_cast<const char*>(embedding.getContainer().data()), sizeof(scalar_type)*embedding.getContainer().size());
      }
    }

    hdi::utils::secureLog(&log, "");
    if (data_loading_time != 0.f) {
      hdi::utils::secureLogValue(&log, "Data loading (s)", data_loading_time);
    }
    hdi::utils::secureLogValue(&log, "Similarities (s)", similarities_comp_time);
    hdi::utils::secureLogValue(&log, "Gradient descent (s)", gradient_desc_comp_time);
    hdi::utils::secureLogValue(&log, "KL computation (s)", kl_divergence_comp_time);
    hdi::utils::secureLogValue(&log, "Data saving (s)", data_saving_time);
    hdi::utils::secureLog(&log, "");
    hdi::utils::secureLogValue(&log, "KL Divergence", KL);

    // Spin until window is closed
    while (!context.shouldClose()) {
      context.poll();
    }

    return EXIT_SUCCESS; // nah app.exec();
  }
  catch (std::logic_error& ex) { std::cerr << "Logic error: " << ex.what() << std::endl; }
  catch (std::runtime_error& ex) { std::cerr << "Runtime error: " << ex.what() << std::endl; }
  catch (std::exception& ex) { std::cerr << "General error: " << ex.what() << std::endl; }
  catch (...) { std::cerr << "An unknown error occurred" << std::endl; }
  return EXIT_FAILURE;
}