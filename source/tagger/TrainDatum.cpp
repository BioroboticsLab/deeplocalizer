#include <QPainter>
#include <boost/filesystem.hpp>
#include <caffe/util/io.hpp>

#include "utils.h"
#include "TrainDatum.h"

namespace  deeplocalizer {


namespace io = boost::filesystem;

TrainDatum::TrainDatum(const std::string &image_filename, const Tag &tag, cv::Mat mat,
              cv::Point2i translation, double rotation_angle) :
        _original_image_filename(image_filename),
        _tag(tag),
        _mat(mat),
        _translation(translation),
        _rotation_angle(rotation_angle)
{
}

TrainDatum::TrainDatum(const std::string &image_filename, const Tag &tag,
                     cv::Mat mat) :
        _original_image_filename(image_filename),
        _tag(tag),
        _mat(mat),
        _translation(cv::Point2i(0, 0)),
        _rotation_angle(0)
{
}

const std::string TrainDatum::filename() const {
    io::path path(_original_image_filename);
    io::path extension = path.extension();
    path.replace_extension("");
    std::stringstream ss;
    cv::Rect bb = _tag.getBoundingBox();
    long angle = std::lround(_rotation_angle);

    std::string right_or_wrong;
    if(_tag.isTag()) {
        right_or_wrong = "_right";
    } else {
        right_or_wrong = "_wrong";
    }
    ss << "_bx" << bb.x << "_by" << bb.y <<
            "_tx" << _translation.x << "_ty" << _translation.y <<
            "_a" << angle << right_or_wrong;
    return path.string() + ss.str() + extension.string();
}

void TrainDatum::draw(QPainter &painter) const {
    static const int line_width = 1;
    painter.save();
    painter.translate(_tag.center().x, _tag.center().y);
    painter.rotate(_rotation_angle);
    painter.translate(-_tag.center().x, -_tag.center().y);
    painter.translate(_translation.x, _translation.y);
    _tag.draw(painter, line_width);
    painter.restore();
}
caffe::Datum TrainDatum::toCaffe() const {
    caffe::Datum datum;
    ASSERT(_mat.cols == TAG_WIDTH, "Expected _mat.cols to be equal " << TAG_WIDTH);
    ASSERT(_mat.rows == TAG_HEIGHT, "Expected _mat.rows to be equal " << TAG_HEIGHT);
    ASSERT(_mat.channels() == 1, "Expected one channel");
    ASSERT(_mat.type() == CV_8U, "Expected one channel");

    std::vector<uchar> buf;
    cv::imencode(".jpeg", _mat, buf);
    datum.set_channels(1);
    datum.set_width(TAG_WIDTH);
    datum.set_height(TAG_HEIGHT);
    datum.set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
    datum.set_label(_tag.isTag());
    datum.set_encoded(true);
    return datum;
}
}
