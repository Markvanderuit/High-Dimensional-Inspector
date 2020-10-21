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


#ifndef HD_JOINT_PROBABILITY_GENERATOR_INL
#define HD_JOINT_PROBABILITY_GENERATOR_INL

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/utils/math_utils.h"
#include "hdi/utils/log_helper_functions.h"
#include "hdi/utils/scoped_timers.h"
#include <random>
#include <chrono>
#include <unordered_set>
#include <numeric>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>

#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif

#pragma warning( push )
#pragma warning( disable : 4267)
#pragma warning( push )
#pragma warning( disable : 4291)
#pragma warning( push )
#pragma warning( disable : 4996)
#pragma warning( push )
#pragma warning( disable : 4018)
#pragma warning( push )
#pragma warning( disable : 4244)
#include "flann/flann.h"
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )
#pragma warning( pop )

namespace hdi{
  namespace dr{
  /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Parameters::Parameters():
      _perplexity(30),
      _perplexity_multiplier(3),
      _num_trees(4),
      _num_checks(1024)
    {}

  /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::Statistics():
      _total_time(0),
      _trees_construction_time(0),
      _aknn_time(0),
      _distribution_time(0)
    {}

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::reset(){
      _total_time = 0;
      _trees_construction_time = 0;
      _aknn_time = 0;
      _distribution_time = 0;
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Statistics::log(utils::AbstractLog* logger)const{
      utils::secureLog(logger,"\n-------- HD Joint Probability Generator Statistics -----------");
      utils::secureLogValue(logger,"Total time",_total_time);
      utils::secureLogValue(logger,"\tTrees construction time",_trees_construction_time,true,1);
      utils::secureLogValue(logger,"\tAKNN time",_aknn_time,true,3);
      utils::secureLogValue(logger,"\tDistributions time",_distribution_time,true,2);
      utils::secureLog(logger,"--------------------------------------------------------------\n");
    }


  /////////////////////////////////////////////////////////////////////////

    template <typename scalar, typename sparse_scalar_matrix>
    HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::HDJointProbabilityGenerator():
      _logger(nullptr)
    {
  
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeJointProbabilityDistribution(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger,"Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int>      indices;
      

      { // TODO Remove FAISS test
        using uint = unsigned;
        const uint n = num_dps;
        const uint d = num_dim;
        const uint k = 3 * static_cast<uint>(params._perplexity) + 1;
        const float *points = high_dimensional_data;

        distances_squared.resize(n * k);
        std::vector<faiss::Index::idx_t> faissIndices(n * k);

        {
          // utils::ScopedTimer<float, utils::Seconds> timer(_time_knn);
          utils::secureLog(_logger, "Performing FAISS test");
        
          // Nr. of inverted lists used by FAISS. O(sqrt(n)) is apparently reasonable
          // src: https://github.com/facebookresearch/faiss/issues/112
          uint nLists = 4 * static_cast<uint>(std::sqrt(n)); 

          // Use a single GPU device. For now, just grab device 0 and pray
          faiss::gpu::StandardGpuResources faissResources;
          faiss::gpu::GpuIndexIVFFlatConfig faissConfig;
          faissConfig.device = 0;
          faissConfig.indicesOptions = faiss::gpu::INDICES_32_BIT;
          faissConfig.flatConfig.useFloat16 = true;
        
          // Construct search index
          // Inverted file flat list gives accurate results at significant memory overhead.
          faiss::gpu::GpuIndexIVFFlat faissIndex(
            &faissResources,
            d, 
            nLists,
            faiss::METRIC_L2, 
            faissConfig
          );
          faissIndex.setNumProbes(12);
          faissIndex.train(n, points);
          faissIndex.add(n, points);

          // Perform actual search
          // Store results device cide in cuKnnSquaredDistances, cuKnnIndices, as the
          // rest of construction is performed on device as well.
          faissIndex.search(
            n,
            points,
            k,
            distances_squared.data(),
            faissIndices.data()
          );
        
          // Tell FAISS to bugger off
          faissIndex.reset();
          faissIndex.reclaimMemory();
        }

        // Cast data
        indices = std::vector<int>(std::begin(faissIndices), std::end(faissIndices));
      } // TODO Remove FAISS test
      const unsigned nn = params._perplexity*params._perplexity_multiplier + 1;
      computeGaussianDistributions(distances_squared,indices,distribution,params);
      symmetrize(distribution);

      std::cout << "CPU" << '\n';
      unsigned i = 0;
      for (auto &p_ij : distribution[0]) {
        std::cout << i << '\t' << p_ij.first << '\t' << p_ij.second << '\n';
        i++;
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger,"Computing the HD joint probability distribution...");
      distribution.resize(num_dps);

      std::vector<scalar_type>  distances_squared;
      std::vector<int>      indices;

      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared,indices,distribution,params);
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributions(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, std::vector<scalar_type>& probabilities, std::vector<int>& indices, Parameters params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._total_time);

      hdi::utils::secureLog(_logger,"Computing the HD joint probability distribution...");

      std::vector<scalar_type>  distances_squared;
      computeHighDimensionalDistances(high_dimensional_data, num_dim, num_dps, distances_squared, indices, params);
      computeGaussianDistributions(distances_squared,indices,probabilities,params);
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeHighDimensionalDistances(scalar_type* high_dimensional_data, unsigned int num_dim, unsigned int num_dps, std::vector<scalar_type>& distances_squared, std::vector<int>& indices, Parameters& params){
      hdi::utils::secureLog(_logger,"Computing nearest neighborhoods...");
      flann::Matrix<scalar_type> dataset  (high_dimensional_data,num_dps,num_dim);
      flann::Matrix<scalar_type> query  (high_dimensional_data,num_dps,num_dim);

      flann::Index<flann::L2<scalar_type> > index(dataset, flann::KDTreeIndexParams(params._num_trees));
      const unsigned int nn = params._perplexity * params._perplexity_multiplier + 1;
      distances_squared.resize(num_dps*nn);
      indices.resize(num_dps*nn);
      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._trees_construction_time);
        index.buildIndex();
      }
      {
        utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._aknn_time);
        flann::Matrix<int> indices_mat(indices.data(), query.rows, nn);
        flann::Matrix<scalar_type> dists_mat(distances_squared.data(), query.rows, nn);
        flann::SearchParams flann_params(params._num_checks);
        flann_params.cores = 0; //all cores
        index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int>& indices, sparse_scalar_matrix& distribution, Parameters& params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger,"Computing joint-probability distribution...");
      const int n = distribution.size();

      const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;

      scalar_vector_type temp_vector(distances_squared.size(),0);
      
      #pragma omp parallel for
      for(int j = 0; j < n; ++j){
        const auto sigma =  utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
                  distances_squared.begin() + j*nn, //check squared
                  distances_squared.begin() + (j + 1)*nn,
                  temp_vector.begin() + j*nn,
                  temp_vector.begin() + (j + 1)*nn,
                  params._perplexity,
                  200,
                  1e-5,
                  0
                );
      }

      for(int j = 0; j < n; ++j){
        for(int k = 1; k < nn; ++k){
          const unsigned int i = j*nn+k;
          distribution[j][indices[i]] = temp_vector[i];
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeGaussianDistributions(const std::vector<scalar_type>& distances_squared, const std::vector<int>& indices, std::vector<scalar_type>& probabilities, Parameters& params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger,"Computing joint-probability distribution...");

      const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
      const int n = indices.size() / nn;
      
      #pragma omp parallel for
      for(int j = 0; j < n; ++j){
        const auto sigma =  utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
                  distances_squared.begin() + j*nn, //check squared
                  distances_squared.begin() + (j + 1)*nn,
                  probabilities.begin() + j*nn,
                  probabilities.begin() + (j + 1)*nn,
                  params._perplexity,
                  200,  // 200
                  1e-5,
                  0
                );
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::symmetrize(sparse_scalar_matrix& distribution){
      const int n = distribution.size();
      for(int j = 0; j < n; ++j){
        for(auto& e: distribution[j]){
          const unsigned int i = e.first;
          scalar new_val = 0.5 * (distribution[j][i] + distribution[i][j]);
          distribution[j][i] = new_val;
          distribution[i][j] = new_val;
        }
      }
    }

    template <typename scalar, typename sparse_scalar_matrix>
    void HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::computeProbabilityDistributionsFromDistanceMatrix(const std::vector<scalar_type>& squared_distance_matrix, unsigned int num_dps, sparse_scalar_matrix& distribution, Parameters params){
      utils::ScopedTimer<scalar_type, utils::Seconds> timer(_statistics._distribution_time);
      utils::secureLog(_logger,"Computing joint-probability distribution...");
      const int n = num_dps;
      const unsigned int nn = num_dps;

      scalar_vector_type temp_vector(num_dps*num_dps,0);

      distribution.clear();
      distribution.resize(n);

      #pragma omp parallel for
      for(int j = 0; j < n; ++j){
        const auto sigma =  utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
                  squared_distance_matrix.begin() + j*nn, //check squared
                  squared_distance_matrix.begin() + (j + 1)*nn,
                  temp_vector.begin() + j*nn,
                  temp_vector.begin() + (j + 1)*nn,
                  params._perplexity,
                  200,
                  1e-5,
                  j
                );
      }

      for(int j = 0; j < n; ++j){
        for(int k = 0; k < nn; ++k){
          const unsigned int i = j*nn+k;
          distribution[j][k] = temp_vector[i];
        }
      }
    }

///////////////////////////////////////////////////////////////////////////////////7


  }
}
#endif 

