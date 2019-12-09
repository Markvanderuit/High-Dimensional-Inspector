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

#include "hdi/visualization/pointcloud_drawer_fixed_color.h"
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include "opengl_helpers.h"
#include "hdi/utils/assert_by_exception.h"

#define GLSL(version, shader)  "#version " #version "\n" #shader

namespace hdi{
  namespace viz{

    PointcloudDrawerFixedColor::PointcloudDrawerFixedColor():
      _program(nullptr),
      _vshader(nullptr),
      _fshader(nullptr),
      _color(qRgb(0,150,255)),
      _initialized(false),
      _point_size(2.5),
      _alpha(0.5)
    { }

    void PointcloudDrawerFixedColor::initialize(QGLContext* context){
      const char *vsrc = GLSL(130,
          in highp vec4 pos_attr;  

          uniform highp mat4 matrix;          
          uniform highp float alpha;
          uniform highp vec4 color;

          out lowp vec4 col;    

          void main() {
            gl_Position = matrix * pos_attr;    
            col = vec4(color.xyz, alpha);
          }                      
        );

      const char *fsrc = GLSL(130,
          in lowp vec4 col;            
          void main() {      
            gl_FragColor = col;
            // gl_FragColor = vec4(vec3(col) * (1.f - gl_FragCoord.z), col.a);
          }                      
        );

      _vshader = std::unique_ptr<QGLShader>(new QGLShader(QGLShader::Vertex, context));
      _fshader = std::unique_ptr<QGLShader>(new QGLShader(QGLShader::Fragment, context));

      _vshader->compileSourceCode(vsrc);
      _fshader->compileSourceCode(fsrc);

      _program = std::unique_ptr<QGLShaderProgram>(new QGLShaderProgram(context));
      _program->addShader(_vshader.get());
      _program->addShader(_fshader.get());
      _program->link();

      _coords_attribute  = _program->attributeLocation("pos_attr");
      _flags_attribute  = _program->attributeLocation("flag_attr");
      _matrix_uniform    = _program->uniformLocation("matrix");
      _color_uniform    = _program->uniformLocation("color");
      _alpha_uniform    = _program->uniformLocation("alpha");
  
      _program->release();
      _initialized = true;
    }

    void PointcloudDrawerFixedColor::draw(const point_type& minb, const point_type& maxb, const QQuaternion& rotation){
      checkAndThrowLogic(_initialized,"Shader must be initilized");
      ScopedCapabilityEnabler blend_helper(GL_BLEND);
      ScopedCapabilityEnabler depth_test_helper(GL_DEPTH_TEST);
      ScopedCapabilityEnabler point_smooth_helper(GL_POINT_SMOOTH);
      ScopedCapabilityEnabler multisample_helper(GL_MULTISAMPLE);
      glDepthFunc(GL_LESS);
      glPointSize(_point_size);

      auto diameter = maxb.distanceToPoint(minb);

      _program->bind();
        QMatrix4x4 matrix;
        matrix.ortho(minb.x(), maxb.x(), minb.y(), maxb.y(), -0.5 * diameter, 0.5f * diameter);

        auto eye = rotation.rotatedVector(QVector3D(0, 0, -1));
        auto up = rotation.rotatedVector(QVector3D(0, 1, 0));
        matrix.lookAt(eye, QVector3D(0, 0, 0), up);

        _program->setUniformValue(_matrix_uniform, matrix);
        _program->setUniformValue(_color_uniform, _color);
        _program->setUniformValue(_alpha_uniform, _alpha);

        QOpenGLFunctions glFuncs(QOpenGLContext::currentContext());
        QOpenGLExtraFunctions glExtraFuncs(QOpenGLContext::currentContext());
        glFuncs.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);

        _program->enableAttributeArray(_coords_attribute);
        glFuncs.glVertexAttribPointer(_coords_attribute, 3, GL_FLOAT, GL_FALSE, 0, _embedding);

        glDrawArrays(GL_POINTS, 0, _num_points);
      _program->release();
    }

    void PointcloudDrawerFixedColor::setData(const scalar_type* embedding, int num_points){
      _embedding = embedding;
      _num_points = num_points;
    }
  }
}
