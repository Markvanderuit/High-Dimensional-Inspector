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

#pragma once

#include <iostream>
#include <unordered_set>
#include <vector>
#include <nanoflann/nano_flann.h>
#include "hdi/utils/assert_by_exception.h"
#include "hdi/data/vptree.h"
#include "hdi/dimensionality_reduction/tsne_parameters.h"

namespace hdi::dr {
  class EmbeddingNanoFlannAdaptor {
  public:
    EmbeddingNanoFlannAdaptor(const std::vector<float> &embedding, const TsneParameters &params) 
    : _embedding(embedding),
      _params(params)
    { }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
      return _params.n;
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t) const {
      float dist = 0.f;
      for(int d = 0; d < _params.nLowDimensions; ++d){
        const float v = p1[d] - _embedding[idx_p2 * _params.nLowDimensions + d];
        dist += v * v;
      }
      return dist;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {
      return _embedding[idx * _params.nLowDimensions + dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
      return false;
    }

  private:
    const TsneParameters _params;
    const std::vector<float> &_embedding;
  };

  void computePrecisionRecall(const std::vector<float> &data,
                              const std::vector<float> &embedding,
                              const TsneParameters &params,
                              const std::vector<unsigned int>& pnts_to_evaluate, 
                              std::vector<float>& precision, 
                              std::vector<float>& recall, 
                              unsigned int K)
  {
    checkAndThrowLogic(pnts_to_evaluate.size() > 0,"computePrecisionRecall: At least one point must be evaluated");
    checkAndThrowLogic(K > 1,"computePrecisionRecall: K must be higher than 1");


    std::cout << '\t' << "Building KD Tree index" << std::endl;
    // Bloody hell what a line. I swear I didn't write this
    // KD Tree with nanoflann for KNN in the embedding
    EmbeddingNanoFlannAdaptor emb_adaptor(embedding, params);
    typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, EmbeddingNanoFlannAdaptor>,
      EmbeddingNanoFlannAdaptor
    > embedding_kd_tree_type;
    embedding_kd_tree_type emb_index(params.nLowDimensions, emb_adaptor);
    emb_index.buildIndex();

    precision = std::vector<float>(K, 0);
    recall = std::vector<float>(K, 0);
    
    // temp
    std::cout << '\t' << "Building VP Tree index" << std::endl;
    VpTree<DataPoint, euclidean_distance<float>> tree;
    std::vector<DataPoint> obj_X(params.n, DataPoint(params.nHighDimensions, -1, data.data()));
    #pragma omp parallel for
    for (int i = 0; i < params.n; i++) {
      obj_X[i] = DataPoint(params.nHighDimensions, i, data.data() + i * params.nHighDimensions);
    }
    tree.create(obj_X);

    std::cout << '\t' << "Evaluating points" << std::endl;
    #pragma omp parallel for
    for(int i = 0; i < pnts_to_evaluate.size(); ++i) {
      unsigned int id = pnts_to_evaluate[i];
      std::vector<DataPoint> indices_hd;
      std::vector<float> distances_hd;
      auto& obj = obj_X[id];
      tree.search(obj_X[id], K + 1, &indices_hd, &distances_hd);

      nanoflann::KNNResultSet<float>  emb_result_set(K+1);
      std::vector<size_t> indices_emb(K+1,0);
      std::vector<float> distances_emb(K+1,0);
      emb_result_set.init(indices_emb.data(),distances_emb.data());
      emb_index.findNeighbors(emb_result_set, embedding.data() + (id * params.nLowDimensions), nanoflann::SearchParams(10));

      std::unordered_set<unsigned int> hd_id_set;
      for(int j = 0; j < K; ++j) {
        hd_id_set.insert(indices_hd[1+j].index());
      }

      for(int k = 1; k <= K; ++k) {
        unsigned int num_positive(0);
        for(int j = 0; j < k; ++j){
          if(hd_id_set.find(indices_emb[1+j]) != hd_id_set.end()){
            ++num_positive;
          }
        }
        
        float precision_val = float(num_positive) / k;
        float recall_val = float(num_positive) / K;

        #pragma omp critical
        {
          precision[k-1] += precision_val;
          recall[k-1] += recall_val;
        }
      }
    }

    const float div = 1.0f / static_cast<float>(pnts_to_evaluate.size());

    #pragma omp parallel for
    for (int i = 0; i < precision.size(); i++) {
      precision[i] *= div;
    }

    #pragma omp parallel for
    for (int i = 0; i < recall.size(); i++) {
      recall[i] *= div;
    }
  }

 /*  template <typename scalar_type>
  void computePrecisionRecall(const data::PanelData<scalar_type>& panel_data, const data::Embedding<scalar_type>& embedding, const std::vector<unsigned int>& pnts_to_evaluate, const std::vector<unsigned int>& emb_id_to_panel_data_id, std::vector<scalar_type>& precision, std::vector<scalar_type>& recall, unsigned int K){
    checkAndThrowLogic(pnts_to_evaluate.size()>0,"computePrecisionRecall: At least one point must be evaluated");
    checkAndThrowLogic(K>1,"computePrecisionRecall: K must be higher than 1");

    //KD Tree with nanoflann for KNN in the embedding
    EmbeddingNanoFlannAdaptor<scalar_type> emb_adaptor(embedding);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<scalar_type, EmbeddingNanoFlannAdaptor<scalar_type> >,EmbeddingNanoFlannAdaptor<scalar_type>> embedding_kd_tree_type;
    embedding_kd_tree_type  emb_index(embedding.numDimensions(), emb_adaptor);
    emb_index.buildIndex();

    precision.clear();
    recall.clear();
    precision.resize(K,0);
    recall.resize(K,0);

    //temp
    VpTree<DataPoint, euclidean_distance<float> >tree;
    std::vector<DataPoint> obj_X(panel_data.numDataPoints(), DataPoint(panel_data.numDimensions(), -1, panel_data.getData().data()));
    {
      int n = 0;
      for(n = 0; n < panel_data.numDataPoints(); n++)
        obj_X[n] = DataPoint(panel_data.numDimensions(), n, panel_data.getData().data() + n * panel_data.numDimensions());
      tree.create(obj_X);
    }

    #pragma omp parallel for
    for(int i = 0; i < pnts_to_evaluate.size(); ++i){
      unsigned int id_pd = emb_id_to_panel_data_id[pnts_to_evaluate[i]];
      unsigned int id_em = pnts_to_evaluate[i];

      std::vector<DataPoint> indices_hd;
      std::vector<float> distances_hd;
      tree.search(obj_X[id_pd], K + 1, &indices_hd, &distances_hd);

      nanoflann::KNNResultSet<scalar_type>  emb_result_set(K+1);
      std::vector<size_t>           indices_emb(K+1,0);
      std::vector<scalar_type>        distances_emb(K+1,0);
      emb_result_set.init(indices_emb.data(),distances_emb.data());
      emb_index.findNeighbors(emb_result_set,&(embedding.getContainer()[id_em*embedding.numDimensions()]),nanoflann::SearchParams(10));

      std::unordered_set<unsigned int> hd_id_set;
      for(int j = 0; j < K; ++j){
        hd_id_set.insert(indices_hd[1+j].index());
      }

      for(int k = 1; k <= K; ++k){
        unsigned int num_positive(0);
        for(int j = 0; j < k; ++j){
          if(hd_id_set.find(emb_id_to_panel_data_id[indices_emb[1+j]]) != hd_id_set.end()){
            ++num_positive;
          }
        }
        float precision_val = float(num_positive) / k;
        float recall_val = float(num_positive) / K;
        
        #pragma omp critical
        {
          precision[k-1] += precision_val;
          recall[k-1] += recall_val;
        }
      }
    }

    for(auto& v: precision){
      v /= pnts_to_evaluate.size();
    }

    for(auto& v: recall){
      v /= pnts_to_evaluate.size();
    }
  } */
}
