#pragma once

#include <opencv2/core.hpp>

namespace deeplocalizer {

    const int TAG_WIDTH = 64;
    const int TAG_HEIGHT = 64;
    const cv::Point2i TAG_CENTER{TAG_WIDTH / 2, TAG_HEIGHT / 2};
    const cv::Size2i TAG_SIZE{TAG_WIDTH, TAG_HEIGHT};

    static const int MAX_TRANSLATION = TAG_WIDTH / 8;
    static const int MIN_TRANSLATION = -MAX_TRANSLATION;
    static const int MIN_AROUND_WRONG = TAG_WIDTH / 2;
    static const int MAX_AROUND_WRONG = TAG_WIDTH / 2 + 80;
    static const double TAGINESS_STD = 1./MAX_TRANSLATION;
    static const double RATIO_AROUND_TO_UNIFORM_DEFAULT = 0.2;
    static const double RATIO_TRUE_TO_FALSE_SAMPLES_DEFAULT = 1.;


    cv::Mat getSubimage(const cv::Mat & orginal, cv::Rect box,
                        unsigned int additional_border=0);

    inline cv::Rect tagBoxForCenter(const cv::Point2i p) {
        return cv::Rect(cv::Point2i(p.x - TAG_WIDTH/2, p.y - TAG_WIDTH/2), TAG_SIZE);
    }
}
