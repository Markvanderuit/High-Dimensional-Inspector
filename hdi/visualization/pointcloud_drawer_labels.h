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

#ifndef POINTCLOUD_DRAWER_LABELS_H
#define POINTCLOUD_DRAWER_LABELS_H

#include <QGLShaderProgram>
#include <QGLShader>
#include <stdint.h>
#include <memory>
#include "hdi/visualization/abstract_pointcloud_drawer.h"

namespace hdi{
  namespace viz{

    //! Drawer for the single-color visualization of a pointcloud
    /*!
      Drawer for the single-color visualization of a pointcloud
      \note Provides a QGLWidget with trackball interaction on a pointcloud
      \author Mark van de Ruit
    */
    class PointcloudDrawerLabels : public AbstractPointcloudDrawer {
    public:
      PointcloudDrawerLabels();
      //! Draw on canvas
      virtual void draw(const point_type& minb, const point_type& maxb, const QQuaternion& rotation);
      
      //! Set the data to be drawn
      void setData(const scalar_type* embedding,  
                   const unsigned int* labels, 
                   const std::map<unsigned int, QColor>& palette, 
                   int num_points);
      void setData(const scalar_type* embedding,  
                   const scalar_type* labels, 
                   const std::map<scalar_type, QColor>& palette, 
                   int num_points);

      virtual void initialize(QGLContext* context);

      void setPointSize(scalar_type point_size){_point_size = point_size;}
      void setAlpha(scalar_type alpha){_alpha = alpha;}
      
      scalar_type pointSize()const{ return _point_size;}
      scalar_type alpha()const{return _alpha;}

    private:
      std::unique_ptr<QGLShaderProgram> _program;
      std::unique_ptr<QGLShader> _vshader;
      std::unique_ptr<QGLShader> _fshader;
      
      GLuint _coords_attribute;
      GLuint _colors_attribute;
      GLuint _matrix_uniform;
      GLuint _alpha_uniform;

      const scalar_type* _embedding;
      std::vector<scalar_type> _colors;
      const flag_type* _flags;

      int _num_points;
      bool _initialized;

      scalar_type _point_size;
      scalar_type _alpha;
    };
  }
}

#endif
