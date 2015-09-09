
#ifndef DEEP_LOCALIZER_QT_HELPER_H
#define DEEP_LOCALIZER_QT_HELPER_H

#include <QImage>
#include <QPixmap>
#include <QDebug>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

Q_DECLARE_METATYPE(cv::Mat)

namespace deeplocalizer {
QImage cvMatToQImage(const cv::Mat &inMat);

inline QPixmap cvMatToQPixmap(const cv::Mat &inMat) {
    return QPixmap::fromImage(cvMatToQImage(inMat));
}
void registerQMetaTypes();
}

#endif //DEEP_LOCALIZER_QT_HELPER_H
