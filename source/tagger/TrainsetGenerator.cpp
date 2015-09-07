#include "TrainsetGenerator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>
#include <thread>

#include "utils.h"
#include "Dataset.h"
#include "DataWriter.h"

namespace deeplocalizer {

const int TrainsetGenerator::MAX_TRANSLATION = TAG_WIDTH / 7;
const int TrainsetGenerator::MIN_TRANSLATION = -TrainsetGenerator::MAX_TRANSLATION;
const int TrainsetGenerator::MIN_AROUND_WRONG = TAG_WIDTH / 2;
const int TrainsetGenerator::MAX_AROUND_WRONG = TAG_WIDTH / 2 + 1000;

TrainsetGenerator::TrainsetGenerator() :
    TrainsetGenerator(std::make_unique<DevNullWriter>())
{}

TrainsetGenerator::TrainsetGenerator(std::unique_ptr<DataWriter> writer)
    :
    _gen(_rd()),
    _angle_dis(0, 360),
    _translation_dis(MIN_TRANSLATION, MAX_TRANSLATION),
    _around_wrong_dis(MIN_AROUND_WRONG, MAX_AROUND_WRONG),
    _writer(std::move(writer)) {
}

TrainsetGenerator::TrainsetGenerator(TrainsetGenerator &&gen) :
    QObject(nullptr),
    _gen(_rd()),
    _angle_dis(std::move(gen._angle_dis)),
    _translation_dis(std::move(gen._translation_dis)),
    _around_wrong_dis(std::move(gen._around_wrong_dis)),
    _writer(std::move(gen._writer)) { }

cv::Mat TrainsetGenerator::randomAffineTransformation(const cv::Point2f & center) {
    static const double scale = 1;
    double angle = _angle_dis(_gen);
    qDebug() << angle;
    return cv::getRotationMatrix2D(center, angle, scale);
}

cv::Mat TrainsetGenerator::rotate(const cv::Mat & src, double degrees) {
    cv::Mat frameRotated;
    int diagonal = int(sqrt(src.cols * src.cols + src.rows * src.rows));
    int offsetX = (diagonal - src.cols) / 2;
    int offsetY = (diagonal - src.rows) / 2;
    cv::Mat targetMat(diagonal, diagonal, src.type(), cv::Scalar(0));
    cv::Point2f src_center(targetMat.cols / 2.0f, targetMat.rows / 2.0f);
    src.copyTo(targetMat.rowRange(offsetY, offsetY + src.rows)
                       .colRange(offsetX, offsetX + src.cols));
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, degrees, 1.0);
    cv::warpAffine(targetMat, frameRotated, rot_mat, targetMat.size());
    return frameRotated;
}

TrainDatum TrainsetGenerator::trainData(const ImageDesc & desc, const Tag &tag,
                                       const cv::Mat & subimage) {
    double angle = _angle_dis(_gen);
    cv::Mat rot_img = rotate(subimage, angle);
    int trans_x, trans_y;
    do {
        trans_x = _translation_dis(_gen);
        trans_y = _translation_dis(_gen);
    } while(abs(trans_x * trans_y) > pow(TAG_WIDTH/3, 2));
    cv::Point2f rot_center(rot_img.cols / 2 + trans_x,
                           rot_img.rows / 2 + trans_y);
    cv::Rect box(int(rot_center.x - TAG_WIDTH / 2),
                 int(rot_center.y - TAG_HEIGHT / 2),
                 TAG_WIDTH, TAG_HEIGHT);
    return TrainDatum(desc.filename, tag, rot_img(box).clone(),
                     cv::Point2i(trans_x, trans_y), angle);
}

void TrainsetGenerator::trueSamples(
        const ImageDesc & desc, const Tag &tag, const cv::Mat & subimage,
        std::vector<TrainDatum> & train_data) {
    if(! tag.isTag()) return;

    for(unsigned int i = 0; i < samples_per_tag; i++) {
        train_data.emplace_back(trainData(desc, tag, subimage));
    }
}

void TrainsetGenerator::trueSamples(const ImageDesc &desc,
                                    std::vector<TrainDatum> &train_data) {
    Image img = Image(desc);
    for(const auto & tag : desc.getTags()) {
        if(tag.isTag()) {
            cv::Mat subimage =  tag.getSubimage(img.getCvMat(), 50);
            trueSamples(desc, tag, subimage, train_data);
        }
    }
}


void TrainsetGenerator::process(const ImageDesc &desc,
                                    std::vector<TrainDatum> &train_data) {
    trueSamples(desc, train_data);
    std::cout << "wrongSamples" << std::endl;
    wrongSamples(desc, train_data);
}
void TrainsetGenerator::wrongSamplesAround(const Tag &tag,
                                           const ImageDesc &desc,
                                           const Image &img,
                                           std::vector<TrainDatum> &train_data) {
    if(not tag.isTag()) return;
    unsigned int samples = 0;
    std::vector<cv::Rect> nearbyBoxes = getNearbyTagBoxes(tag, desc);

    cv::Rect img_rect = cv::Rect(cv::Point(0, 0), img.getCvMat().size());

    ASSERT(samples_per_tag % 4 == 0, "samples_per_tag must be a multiple of four");

    while(samples < samples_per_tag) {
        cv::Rect box = proposeWrongBox(tag);
        bool contains = (img_rect & box).area() == box.area();
        if (contains && intersectsNone(nearbyBoxes, box)) {
            wrongSampleRot90(img, box, train_data);
            samples += 4;
        }
    }
}
void TrainsetGenerator::wrongSamples(const ImageDesc &desc,
                                     std::vector<TrainDatum> &train_data) {
    Image img{desc};
    for(const auto & tag: desc.getTags()) {
        if(tag.isExclude()) continue;
        wrongSamplesAround(tag, desc, img, train_data);
    }
}

std::vector<cv::Rect> TrainsetGenerator::getNearbyTagBoxes(const Tag &tag,
                                                           const ImageDesc &desc) {
    auto center = tag.center();
    cv::Rect nearbyArea{center.x-MAX_AROUND_WRONG-TAG_WIDTH,
                        center.y-MAX_AROUND_WRONG-TAG_HEIGHT,
                        2*MAX_AROUND_WRONG+2*TAG_WIDTH,
                        2*MAX_AROUND_WRONG+2*TAG_WIDTH};
    std::vector<cv::Rect> nearbyBoxes;
    for(const auto & other_tag: desc.getTags()) {
        if((other_tag.getBoundingBox() & nearbyArea).area()) {
            auto center = other_tag.center();
            auto bb = other_tag.getBoundingBox();
            cv::Rect shrinked_bb{
                    center.x - TAG_WIDTH  / 2 + shrinking,
                    center.y - TAG_HEIGHT / 2 + shrinking,
                    bb.width - 2*shrinking,
                    bb.height - 2*shrinking,
            };
            nearbyBoxes.emplace_back(std::move(shrinked_bb));
        }
    }
    return nearbyBoxes;
}

cv::Rect TrainsetGenerator::proposeWrongBox(const Tag &tag) {
    auto center = tag.center();
    auto wrong_center = center + cv::Point2i(wrongAroundCoordinate(),
                                             wrongAroundCoordinate());
    return cv::Rect(wrong_center.x - TAG_WIDTH  / 2,
                    wrong_center.y - TAG_HEIGHT / 2,
                    TAG_WIDTH, TAG_HEIGHT);
}

bool TrainsetGenerator::intersectsNone(std::vector<cv::Rect> &tag_boxes,
                                       cv::Rect wrong_box) {

    return std::all_of(
            tag_boxes.begin(), tag_boxes.end(),
            [&wrong_box](auto & box) {
                return (box & wrong_box).area() == 0;
    });
}

void TrainsetGenerator::wrongSampleRot90(const Image &img,
                                         const cv::Rect &wrong_box,
                                         std::vector<TrainDatum> &train_data) {
    Tag wrong_tag{wrong_box};
    wrong_tag.setType(TagType::NoTag);
    cv::Mat subimage = wrong_tag.getSubimage(img.getCvMat());
    train_data.emplace_back(TrainDatum(img.filename(), wrong_tag, subimage));
    {
        // clockwise 90°
        double angle = 270;
        cv::Mat rot(subimage.size(), subimage.type());
        cv::transpose(subimage, rot);
        cv::flip(rot, rot, 1);
        train_data.emplace_back(
                TrainDatum(img.filename(), wrong_tag, rot, cv::Point2i(0, 0),
                          angle));
    }
    {
        double angle = 90;
        cv::Mat rot(subimage.size(), subimage.type());
        transpose(subimage, rot);
        flip(rot, rot,0); //transpose+flip(0)=CCW
        train_data.emplace_back(
                TrainDatum(img.filename(), wrong_tag, rot, cv::Point2i(0, 0),
                          angle));
    }
    {

        double angle = 180;
        cv::Mat rot(subimage.size(), subimage.type());
        flip(subimage, rot,-1);
        train_data.emplace_back(
                TrainDatum(img.filename(), wrong_tag, rot, cv::Point2i(0, 0),
                          angle));
    }
}

int TrainsetGenerator::wrongAroundCoordinate() {
    int x = _around_wrong_dis(_gen);
    if(rand() % 2) {
        return x;
    } else {
        return -x;
    }
}
TrainsetGenerator TrainsetGenerator::operator=(TrainsetGenerator &&other) {
    return TrainsetGenerator(std::move(other));
}
void TrainsetGenerator::process(const std::vector<ImageDesc> &descs,
                                Dataset::Phase phase) {
    return process(descs.cbegin(), descs.cend(), phase);
}

void TrainsetGenerator::processParallel(const std::vector<ImageDesc> &img_descs,
                                        const Dataset::Phase phase) {
    using Iter = std::vector<ImageDesc>::const_iterator;
    std::vector<std::thread> threads;
    auto fn = std::mem_fn<void(Iter, Iter, const Dataset::Phase)>(&TrainsetGenerator::process<Iter>);

    _start_time = std::chrono::system_clock::now();
    _n_todo = img_descs.size();
    _n_done.store(0);
    printProgress(_start_time, 0);

    size_t n_cpus = size_t(std::thread::hardware_concurrency()*2);
    if (n_cpus == 0) n_cpus = 1;
    size_t per_cpu = img_descs.size() / n_cpus;
    for(size_t i = 0; i < n_cpus && i < img_descs.size(); i++) {
        auto begin = img_descs.cbegin() + per_cpu*i;
        auto end = img_descs.cbegin() + per_cpu*(i+1);
        if (i + 1 == n_cpus) {
            end = img_descs.cend();
        }
        std::cout << "start thread: " << i << std::endl << std::endl;
        threads.emplace_back(
                std::thread(fn, this, begin, end, phase)
        );
    }
    for(auto &t: threads) {
        t.join();
    }

}
void TrainsetGenerator::incrementDone() {
    double done = _n_done.fetch_add(1);
    printProgress(_start_time, (done+1)/_n_todo);
}
}
