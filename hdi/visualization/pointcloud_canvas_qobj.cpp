#include <QtGui>
#include <QtOpenGL>
#include "hdi/visualization/pointcloud_canvas_qobj.h"
#include "hdi/utils/log_helper_functions.h"
#include "opengl_helpers.h"
#include "hdi/utils/assert_by_exception.h"

namespace hdi{
  namespace viz{

     PointcloudCanvas::PointcloudCanvas(QWidget *parent, QGLWidget *shareWidget)
      : QGLWidget(parent, shareWidget),
      _logger(nullptr),
      _verbose(false),
      _bg_top_color(qRgb(255,255,255)),
      _bg_bottom_color(qRgb(255,255,255)),
      _selection_color(qRgb(50,50,50)),
      _embedding_top_right_corner(1,1),
      _embedding_bottom_left_corner(0,0),
      _selection_in_progress(false),
      _far_val(1),
      _near_val(-1) {
        // Request depth buffer
        QGLFormat format;
        format.setDepthBufferSize(24);
        format.setProfile(QGLFormat::CoreProfile);
        setFormat(format);
      }

     PointcloudCanvas::~PointcloudCanvas(){
     }
    
     void PointcloudCanvas::setBackgroundColors(const QColor &top, const QColor &bottom){
      _bg_top_color = top;
      _bg_bottom_color = bottom;
      updateGL();
     }
     void PointcloudCanvas::getBackgroundColors(QColor &top, QColor &bottom)const{
      top  = _bg_top_color;
      bottom = _bg_bottom_color;
     }


    void PointcloudCanvas::initializeGL(){
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_CULL_FACE);
      glEnable(GL_TEXTURE_2D);
      glEnable(GL_POINT_SMOOTH);
      glEnable (GL_BLEND);
      glEnable(GL_MULTISAMPLE);
      glEnable(GL_PROGRAM_POINT_SIZE);
      glPointSize(2.5);
      /*
      GLenum err= glGetError();
      printf("OpenGL error: %d", err);
      while ((err = glGetError()) != GL_NO_ERROR) {
        printf("OpenGL error: %d", err);
      }
      */
    }

    void PointcloudCanvas::paintGL(){
      // A bit overcomplicated... revise it
      utils::secureLog(_logger, "paintGl", _verbose);
      glClearColor(1, 1, 1, 1);
      glClearDepth(1.0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      for (const auto drawer: _drawers ){
        assert(drawer != nullptr);  
        drawer->draw(_embedding_min_bounds, _embedding_max_bounds, _trackball_rotation);
      }
     }

     void PointcloudCanvas::resizeGL(int width, int height)
     {
       glViewport(0,0,width,height);
     }

     void PointcloudCanvas::mousePressEvent(QMouseEvent *event)
     {
      utils::secureLog(_logger,"Mouse clicked...",_verbose);
      if(event->button() == Qt::LeftButton){
        _trackball_in_progress = true;
        _trackball_last_pos = event->pos();

        _selection_first_point = event->pos();
        _selection_second_point = event->pos();
        _selection_in_progress = true;

        emit sgnLeftButtonPressed(getMousePositionInSpace(event->pos()));
      }
      if(event->button() == Qt::RightButton){
        emit sgnRightButtonPressed(getMousePositionInSpace(event->pos()));
      }
      if(event->button() == Qt::MiddleButton){
        updateGL();
      }
     }

     void PointcloudCanvas::mouseMoveEvent(QMouseEvent *event)
     {
       if (!_trackball_in_progress) {
         return;
       }

      auto delta = event->localPos() - _trackball_last_pos;
      auto axis = QVector3D(delta.y(), delta.x(), 0.0).normalized();
      auto angle = std::sqrt(QPointF::dotProduct(delta, delta));
      _trackball_rotation = _trackball_rotation * QQuaternion::fromAxisAndAngle(axis, angle);
      _trackball_last_pos = event->localPos();

      updateGL();
      emit sgnMouseMove(getMousePositionInSpace(event->pos()));

      //  if(_selection_in_progress){
      //   _selection_second_point = event->pos();
      //   updateGL();
      //  }
      //  emit sgnMouseMove(getMousePositionInSpace(event->pos()));

      //  if((event->buttons()&Qt::MidButton) == Qt::MidButton){
      //    QPoint position = event->pos();
      //    QVector2D lens_center(1.*position.x()/width(),1.*(height()-position.y())/height());
      //    updateGL();
      //  }
     }

    void PointcloudCanvas::wheelEvent ( QWheelEvent * event ){
      int numDegrees = event->delta() / 8;
      updateGL();
    }

     void PointcloudCanvas::mouseReleaseEvent(QMouseEvent *event){
      if(event->button() == Qt::LeftButton){
        _trackball_in_progress = false;

        _selection_second_point = event->pos();
        _selection_in_progress = false;
        updateGL();

        auto p0 = getMousePositionInSpace(_selection_first_point);
        auto p1 = getMousePositionInSpace(_selection_second_point);
        emit sgnSelection(point_type(std::min(p0.x(),p1.x()),std::min(p0.y(),p1.y())),
                  point_type(std::max(p0.x(),p1.x()),std::max(p0.y(),p1.y()))
                  );
        emit sgnLeftButtonReleased(getMousePositionInSpace(event->pos()));
      }
      if(event->button() == Qt::RightButton){
        emit sgnRightButtonReleased(getMousePositionInSpace(event->pos()));
      }
      if(event->button() == Qt::MiddleButton){
        updateGL();
      }

      emit clicked();
     }

    void PointcloudCanvas::mouseDoubleClickEvent(QMouseEvent* event){
      if(event->button() == Qt::LeftButton){
        emit sgnLeftDoubleClick(getMousePositionInSpace(event->pos()));
      }
      if(event->button() == Qt::RightButton){
        emit sgnRightDoubleClick(getMousePositionInSpace(event->pos()));
      }
    }

    QVector2D PointcloudCanvas::getMousePositionInSpace(QPoint pnt)const{
      const point_type tr_bl(_embedding_top_right_corner-_embedding_bottom_left_corner);
      return QVector2D(_embedding_bottom_left_corner.x()+float(pnt.x())/width()*tr_bl.x(),
               _embedding_bottom_left_corner.y()+float(height()-pnt.y())/height()*tr_bl.y());
    }

    void PointcloudCanvas::keyPressEvent(QKeyEvent* e){
      emit sgnKeyPressed(e->key());
    }

    void PointcloudCanvas::saveToFile(std::string filename){
      QImage image = grabFrameBuffer();
      image.save(QString::fromStdString(filename));
    }
  }
}
