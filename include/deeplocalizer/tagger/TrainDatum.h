
#ifndef DEEP_LOCALIZER_TRAINDATA_H
#define DEEP_LOCALIZER_TRAINDATA_H

#include <caffe/proto/caffe.pb.h>
#include "Tag.h"
#include "Image.h"
#include "deeplocalizer_tagger.h"

namespace deeplocalizer {

class TrainDatum {
public:
    TrainDatum(const cv::Mat mat, cv::Point2i center, double rotation_angle, double taginess,
               std::string description="");

    void draw(QPainter & painter) const;
    inline const cv::Point2i & center() const {
        return _center;
    }
    inline double rotation_angle() const {
        return _rotation_angle;
    }
    inline double taginess() const {
        return _taginess;
    }
    inline const cv::Mat & mat() const {
        return _mat;
    }
    inline const std::string & description() const {
        return _description;
    }
    caffe::Datum toCaffe() const;

private:
    cv::Mat _mat;
    cv::Point2i _center;
    double _rotation_angle;
    double _taginess;
    std::string _description;
};
}

#endif //DEEP_LOCALIZER_TRAINDATA_H
