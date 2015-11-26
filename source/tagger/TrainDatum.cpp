#include <QPainter>
#include <boost/filesystem.hpp>
#include <caffe/util/io.hpp>

#include "utils.h"
#include "TrainDatum.h"

namespace  deeplocalizer {


namespace io = boost::filesystem;

TrainDatum::TrainDatum(const cv::Mat mat, cv::Point2i center, double rotation_angle, double taginess, std::string description) :
        _mat(mat),
        _center(center),
        _rotation_angle(rotation_angle),
        _taginess(taginess),
        _description(description)
{
}

//const std::string TrainDatum::filename() const {
//    io::path path(_original_image_filename);
//    io::path extension = path.extension();
//    path.replace_extension("");
//    std::stringstream ss;
//    cv::Rect bb = _tag.getBoundingBox();
//    long angle = std::lround(_rotation_angle);
//
//    std::string right_or_wrong;
//    if(_tag.isTag()) {
//        right_or_wrong = "_right";
//    } else {
//        right_or_wrong = "_wrong";
//    }
//    ss << "_bx" << bb.x << "_by" << bb.y <<
//            "_tx" << _center.x << "_ty" << _center.y <<
//            "_a" << angle << right_or_wrong;
//    return path.filename().string() + ss.str() + extension.string();
//}

void TrainDatum::draw(QPainter &painter) const {
    static const int line_width = 1;
    painter.save();
    painter.translate(_center.x, _center.y);
    painter.rotate(_rotation_angle);
    auto bb = tagBoxForCenter(_center);
    QColor color;
    color.setHsvF(120./360, 1, _taginess);
    painter.setPen(QPen(color, line_width));
    QRect rect(bb.x - _center.x, bb.y - _center.y, bb.height, bb.width);
    painter.drawRect(rect);
    QPoint point(-TAG_WIDTH/2, -TAG_WIDTH/3);
    QString s = QString::number(_taginess, 'f', 2);
    painter.drawText(point, s);
    painter.restore();
}
}
