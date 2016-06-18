#ifndef DEEP_LOCALIZER_WHOLEIMAGEWIDGET_H
#define DEEP_LOCALIZER_WHOLEIMAGEWIDGET_H

#include <thread>
#include <QObject>
#include <QWidget>
#include <QScrollArea>
#include <QPainter>
#include <set>

#include <opencv2/core/core.hpp>
#include <boost/optional/optional.hpp>
#include <QtGui/qpainter.h>
#include "Image.h"
#include "qt_helper.h"
#include "PipelineWorker.h"

namespace deeplocalizer {

class Tag;

class ImageDesc;


class WholeImageWidget : public QWidget {
Q_OBJECT

public:
    WholeImageWidget(QScrollArea * parent);
    WholeImageWidget(QScrollArea * parent, cv::Mat mat, std::vector<Tag> * tags);
    WholeImageWidget(QScrollArea * parent,
                     boost::optional<std::pair<cv::Mat, std::vector<Tag> *>> tags);
    void setTags(cv::Mat mat, std::vector<Tag> * tags);
    virtual QSize sizeHint() const;
public slots:
    boost::optional<Tag> createTag(int x, int y);
    void tagProcessed(Tag tag);
    void zoom(double factor);
    void zoomIn();
    void zoomOut();
    void zoomInRelToMouse(QPoint mouse_pos);
signals:
    void imageFinished();
    void changed();
protected:
    void mousePressEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent * event);
    virtual void paintEvent(QPaintEvent *);
private:
    QScrollArea *_parent;
    cv::Mat _mat;
    QPixmap _pixmap;
    QPainter _painter;
    double _scale = 0.8;
    std::vector<Tag> * _tags;
    std::list<Tag> _newly_added_tags;
    std::set<unsigned long> _deleted_Ids;
    PipelineWorker _pipeline_worker;

    boost::optional<Tag &> getTag(int x, int y);
    void findEllipse(Tag &&tag);

    template<typename T>
    void eraseTag(const unsigned long id, T& tags) {
        _deleted_Ids.insert(id);
        tags.erase(std::remove_if(tags.begin(), tags.end(),
                                  [id](auto & t){
                                      return t.id() == id;
                                  }),
                   tags.end()
        );
    }

    virtual ~WholeImageWidget() {
        _pipeline_worker.quit();
    }
};
}

#endif //DEEP_LOCALIZER_WHOLEIMAGEWIDGET_H
