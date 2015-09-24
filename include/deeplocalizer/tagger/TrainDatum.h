
#ifndef DEEP_LOCALIZER_TRAINDATA_H
#define DEEP_LOCALIZER_TRAINDATA_H

#include <caffe/proto/caffe.pb.h>
#include "Tag.h"
#include "Image.h"

namespace deeplocalizer {

class TrainDatum {
public:
    TrainDatum(const std::string &image_filename, const Tag &tag, cv::Mat mat,
                  cv::Point2i translation, double rotation_angle);

    TrainDatum(const std::string & image_filename, const Tag &tag,
              cv::Mat mat);

    const std::string filename() const;
    void draw(QPainter & painter) const;
    const Tag & tag() const {
        return _tag;
    };
    const cv::Point2i & translation() const {
        return _translation;
    };
    double rotation_angle() const {
        return _rotation_angle;
    };
    void setMat(cv::Mat mat) {
        _mat = mat;
    }

    const cv::Mat & mat() const {
        return _mat;
    };
    caffe::Datum toCaffe() const;

private:
    const std::string _original_image_filename;
    const Tag _tag;
    cv::Mat _mat;
    const cv::Point2i _translation;
    const double _rotation_angle;
};
}

#endif //DEEP_LOCALIZER_TRAINDATA_H
