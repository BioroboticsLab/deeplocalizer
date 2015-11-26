#include "deeplocalizer_tagger.h"

cv::Mat deeplocalizer::getSubimage(const cv::Mat &orginal, cv::Rect box,
                                        unsigned int additional_border) {
    box.x -= additional_border;
    box.y -= additional_border;
    box.width += 2*additional_border;
    box.height+= 2*additional_border;
    if(box.x < 0) box.x = 0;
    if(box.y < 0) box.y = 0;
    if(box.width + box.x >= orginal.cols) box.x = orginal.cols - box.width - 1;
    if(box.height + box.y >= orginal.rows) box.y = orginal.rows - box.height - 1;
    return orginal(box).clone();
}
