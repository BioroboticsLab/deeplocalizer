
#include "WholeImageWidget.h"

#include "utils.h"
#include <QPainter>
#include <QScrollArea>
#include <QGuiApplication>
#include <QScrollBar>
#include <QThread>
#include <QMouseEvent>
#include <QtCore/qline.h>
#include "qt_helper.h"

namespace deeplocalizer {

using boost::optional;

using mattags_t = std::pair<cv::Mat, std::vector<Tag> *>;
using opt_mattags_t = boost::optional<mattags_t>;

WholeImageWidget::WholeImageWidget(QScrollArea *parent)
    : WholeImageWidget(parent, boost::optional<mattags_t>())  {}

WholeImageWidget::WholeImageWidget(QScrollArea * parent, cv::Mat mat, std::vector<Tag> * tags) :
        WholeImageWidget(parent, std::pair<cv::Mat, std::vector<Tag> *>(mat, tags))
{ }
WholeImageWidget::WholeImageWidget(QScrollArea * parent,
                                   opt_mattags_t opt_mattags)
    : QWidget(parent)  {
    _parent = parent;
    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    this->resize(this->sizeHint());
    if(opt_mattags) {
        auto & mattags = opt_mattags.get();
        this->setTags(mattags.first, mattags.second);
    }
}


optional<Tag> WholeImageWidget::createTag(int x, int y) {
    if(x < TAG_WIDTH / 2 || y < TAG_HEIGHT / 2 ||
        x > _pixmap.size().width() - TAG_WIDTH / 2 ||
        y > _pixmap.size().height() - TAG_HEIGHT / 2
            ) {
        return optional<Tag>();
    }

    return Tag(cv::Rect(x - TAG_WIDTH / 2, y - TAG_HEIGHT / 2, TAG_WIDTH, TAG_HEIGHT));
}

void WholeImageWidget::tagProcessed(Tag tag) {
    if(tag.isTag() == TagType::NoTag) {
        tag.setType(TagType::IsTag);
    }
    if(_deleted_Ids.find(tag.id()) != _deleted_Ids.end()) {
        return;
    }
    eraseTag(tag.id(), _newly_added_tags);
    _tags->push_back(tag);
    repaint();
}

void WholeImageWidget::paintEvent(QPaintEvent *) {
    _pixmap = cvMatToQPixmap(_mat);
    _painter.begin(this);

    _painter.scale(_scale, _scale);
    _painter.drawPixmap(0, 0, _pixmap);
    for(auto & t: *_tags) {
        t.draw(_painter);
    }
    for(auto & t: _newly_added_tags) {
        t.draw(_painter);
    }
    _painter.end();
}
void adjustScrollBarRelToMouse(QScrollBar *scrollBar, double mouse_rel_in_viewport, double factor)
{
    scrollBar->setValue(int(factor*scrollBar->value()
                        + (factor - 1) * scrollBar->pageStep() * mouse_rel_in_viewport));
}

void adjustScrollBarCenter(QScrollBar *scrollBar, double factor) {
    scrollBar->setValue(int(factor * scrollBar->value()
                            + (factor - 1) * scrollBar->pageStep() / 2));
}

void WholeImageWidget::zoom(double factor) {
    if(_scale > 3 && factor > 1) return;
    if(_scale < 0.15 && factor < 1) return;
    _scale *= factor;

    resize(sizeHint());
    adjustScrollBarCenter(_parent->horizontalScrollBar(), factor);
    adjustScrollBarCenter(_parent->verticalScrollBar(), factor);
    setFixedSize(sizeHint());
}
void WholeImageWidget::zoomIn() {
    zoom(1.25);
}
void WholeImageWidget::zoomOut() {
    zoom(0.8);
}
void WholeImageWidget::zoomInRelToMouse(QPoint mouse_pos) {
    QPoint pos{mouse_pos / _scale};
    if(_scale > 3) return;
    double factor = 1.25;
    auto vert = _parent->verticalScrollBar();
    auto horz = _parent->horizontalScrollBar();
    auto img_size = _pixmap.size();
    QSize viewport(_parent->viewport()->size());
    if (img_size.width() < viewport.width())  viewport.setWidth(img_size.width());
    if (img_size.height() < viewport.height()) viewport.setHeight(img_size.height());
    QSize max_viewport(_pixmap.size() - viewport);
    QPointF scroll_ratio(horz->value() / double(horz->maximum() - horz->minimum()),
                         vert->value() / double(vert->maximum() - vert->minimum()));
    if (std::isnan(scroll_ratio.x()))  scroll_ratio.setX(0);
    if (std::isnan(scroll_ratio.y())) scroll_ratio.setY(0);
    QPointF viewport_origin{
            scroll_ratio.x()*max_viewport.width(),
            scroll_ratio.y()*max_viewport.height()
    };
    QPointF mouse_rel_in_viewport {
            (pos.x() - viewport_origin.x()) / viewport.width(),
            (pos.y() - viewport_origin.y()) / viewport.height(),
    };
    _scale *= factor;
    setFixedSize(sizeHint());
    resize(sizeHint());
    adjustScrollBarRelToMouse(_parent->horizontalScrollBar(), mouse_rel_in_viewport.x(), factor);
    adjustScrollBarRelToMouse(_parent->verticalScrollBar(), mouse_rel_in_viewport.y(), factor);
}

void WholeImageWidget::wheelEvent(QWheelEvent * event) {
    if (event->delta() > 0) {
        zoomInRelToMouse(event->pos());
    } else if (event->delta() < 0) {
        zoomOut();
    } else {
        return;
    }
}

void WholeImageWidget::mousePressEvent(QMouseEvent * event) {
    auto pos = event->pos() / _scale;
    boost::optional<Tag> opt_tag = getTag(pos.x(), pos.y());
    if (opt_tag) {
        eraseTag(opt_tag.get().id(), *_tags);
        eraseTag(opt_tag.get().id(), _newly_added_tags);
    } else {
        auto modifier = QGuiApplication::queryKeyboardModifiers();
        opt_tag = createTag(pos.x(), pos.y());
        if(!opt_tag) return;
        auto tag = opt_tag.get();
        if (modifier.testFlag(Qt::ControlModifier)) {
            tag.setType(TagType::Exclude);
        } else if (modifier.testFlag(Qt::AltModifier)) {
            tag.setType(TagType::BeeWithoutTag);
        }
        _tags->push_back(tag);
    }
    emit changed();
    repaint();
}

boost::optional<Tag> WholeImageWidget::getTag(int x, int y) {
    auto getTagIfContainsPoint = [x, y](auto & tags) {
        cv::Point point(x, y);
        for(auto & tag : tags) {
            if(tag.getBoundingBox().contains(point)) {
                return optional<Tag>(tag);
            }
        }
        return optional<Tag>();
    };
    boost::optional<Tag> tag = getTagIfContainsPoint(*_tags);
    if (tag) {
        return tag;
    }
    return getTagIfContainsPoint(_newly_added_tags);
}
void WholeImageWidget::setTags(cv::Mat mat, std::vector<Tag> * tags) {
    _mat = mat;
    _pixmap = cvMatToQPixmap(mat);
    _tags = tags;
    setFixedSize(sizeHint());
}

QSize WholeImageWidget::sizeHint() const {
    int width = int(_mat.cols*_scale);
    int height = int(_mat.rows*_scale);
    return QSize(width, height);
}
}
