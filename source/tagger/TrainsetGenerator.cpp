#include "TrainsetGenerator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>
#include <thread>

#include "utils.h"
#include "Dataset.h"
#include "DataWriter.h"

namespace deeplocalizer {

const int TrainsetGenerator::MAX_TRANSLATION = TAG_WIDTH / 8;
const int TrainsetGenerator::MIN_TRANSLATION = -TrainsetGenerator::MAX_TRANSLATION;
const int TrainsetGenerator::MIN_AROUND_WRONG = TAG_WIDTH / 2;
const int TrainsetGenerator::MAX_AROUND_WRONG = TAG_WIDTH / 2 + 80;
const double TrainsetGenerator::RATIO_AROUND_TO_UNIFORM = 0.2;
const double TrainsetGenerator::RATIO_TRUE_TO_FALSE_SAMPLES = 1;

TrainsetGenerator::TrainsetGenerator() :
    TrainsetGenerator(std::make_unique<DevNullWriter>())
{}

TrainsetGenerator::TrainsetGenerator(std::unique_ptr<DataWriter> writer)
    :
    _random_gen(_rd()),
    _angle_dis(0, 360),
    _translation_dis(MIN_TRANSLATION, MAX_TRANSLATION),
    _around_wrong_dis(MIN_AROUND_WRONG, MAX_AROUND_WRONG),
    _writer(std::move(writer)) {
}

TrainsetGenerator::TrainsetGenerator(TrainsetGenerator &&gen) :
    QObject(nullptr),
    _random_gen(_rd()),
    _angle_dis(std::move(gen._angle_dis)),
    _translation_dis(std::move(gen._translation_dis)),
    _around_wrong_dis(std::move(gen._around_wrong_dis)),
    _writer(std::move(gen._writer)) { }

cv::Mat TrainsetGenerator::randomAffineTransformation(const cv::Point2f & center) {
    static const double scale = 1;
    double angle = _angle_dis(_random_gen);
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
    double angle = _angle_dis(_random_gen);
    cv::Mat rot_img = rotate(subimage, angle);
    int trans_x, trans_y;
    do {
        trans_x = _translation_dis(_random_gen);
        trans_y = _translation_dis(_random_gen);
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
            static size_t border = TAG_WIDTH / 2;
            cv::Mat subimage =  tag.getSubimage(img.getCvMat(), border);
            trueSamples(desc, tag, subimage, train_data);
        }
    }
}

std::vector<cv::Rect> getAllBoxes(const std::vector<Tag> & tags) {
    std::vector<cv::Rect>  boxes;
    for (auto & tag : tags) {
        boxes.emplace_back(tag.getBoundingBox());
    }
    return boxes;
}

void TrainsetGenerator::process(const ImageDesc &desc,
                                    std::vector<TrainDatum> &train_data) {
    trueSamples(desc, train_data);
    wrongSamples(desc, train_data);
}

void TrainsetGenerator::wrongSamplesAroundTag(const Tag &tag,
                                           const ImageDesc &desc,
                                           const Image &img,
                                           std::vector<TrainDatum> &train_data) {
    if(not tag.isTag()) return;
    std::vector<cv::Rect> nearbyBoxes = getNearbyTagBoxes(tag, desc);
    cv::Rect img_rect = cv::Rect(cv::Point(0, 0), img.getCvMat().size());
    double to_sample = floor(_avg_samples_per_tag);
    _samples_around_err += _avg_samples_per_tag - to_sample;
    if (_samples_around_err > 1) {
        to_sample += floor(_samples_around_err);
        _samples_around_err -= floor(_samples_around_err);
    }
    for(size_t samples = 0; samples < to_sample; ) {
        cv::Rect box = proposeWrongBoxAround(tag);
        bool contains = (img_rect & box).area() == box.area();
        if (contains && intersectsNone(nearbyBoxes, box)) {
            Tag wrong_tag{box};
            wrong_tag.setType(TagType::NoTag);
            cv::Mat subimage = wrong_tag.getSubimage(img.getCvMat());
            TrainDatum datum(img.filename(), wrong_tag, subimage, cv::Point2i(0, 0), 0);
            train_data.push_back(datum);
            samples++;
        }
    }
}
size_t countTrueTags(const std::vector<Tag> & tags) {
    return std::count_if(tags.cbegin(), tags.cend(), [](const Tag & tag) {
        return not tag.isExclude();
    });
}

void TrainsetGenerator::wrongSamplesUniform(
            const ImageDesc &desc,  const Image &img,
            std::vector<TrainDatum> &train_data) {
    const auto allBoxes = getAllBoxes(desc.getTags());
    std::uniform_int_distribution<> x_dis(0, img.getCvMat().cols);
    std::uniform_int_distribution<> y_dis(0, img.getCvMat().rows);
    const cv::Rect img_rect = cv::Rect(cv::Point(0, 0), img.getCvMat().size());
    const size_t nb_tags = countTrueTags(desc.getTags());
    const size_t to_sample = static_cast<size_t>(
                round(nb_tags * samples_per_tag / RATIO_TRUE_TO_FALSE_SAMPLES / (1+RATIO_AROUND_TO_UNIFORM)));
    size_t samples = 0;
    while(samples < to_sample) {
        cv::Rect box(x_dis(_random_gen), y_dis(_random_gen),
                     TAG_WIDTH, TAG_HEIGHT);
        bool contains = (img_rect & box).area() == box.area();
        if (contains && intersectsNone(allBoxes, box)) {
            Tag wrong_tag{box};
            wrong_tag.setType(TagType::NoTag);
            cv::Mat subimage = wrong_tag.getSubimage(img.getCvMat());
            TrainDatum datum(img.filename(), wrong_tag, subimage, cv::Point2i(0, 0), 0);
            train_data.push_back(datum);
            samples++;
        }
    }
}

void TrainsetGenerator::wrongSamplesAround(
        const ImageDesc &desc, const Image &img,
        std::vector<TrainDatum> &train_data) {

    const size_t nb_tags = countTrueTags(desc.getTags());
    const double total_to_sample = nb_tags * samples_per_tag * RATIO_AROUND_TO_UNIFORM / (1+RATIO_AROUND_TO_UNIFORM)
            / RATIO_TRUE_TO_FALSE_SAMPLES;
    _avg_samples_per_tag = total_to_sample / nb_tags;
    _samples_around_err = 0;
    for(const auto & tag: desc.getTags()) {
        if(tag.isExclude()) continue;
        wrongSamplesAroundTag(tag, desc, img, train_data);
    }
}
void TrainsetGenerator::wrongSamples(const ImageDesc &desc,
                                     std::vector<TrainDatum> &train_data) {
    Image img{desc};
    wrongSamplesAround(desc, img, train_data);
    wrongSamplesUniform(desc, img, train_data);
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
            cv::Rect bb{
                center.x - TAG_WIDTH  / 2,
                center.y - TAG_HEIGHT / 2,
                TAG_HEIGHT,
                TAG_WIDTH,
            };
            nearbyBoxes.emplace_back(bb);
        }
    }
    return nearbyBoxes;
}

cv::Rect TrainsetGenerator::proposeWrongBoxAround(const Tag &tag) {
    auto center = tag.center();
    auto wrong_center = center + cv::Point2i(wrongAroundCoordinate(),
                                             wrongAroundCoordinate());
    return cv::Rect(wrong_center.x - TAG_WIDTH  / 2,
                    wrong_center.y - TAG_HEIGHT / 2,
                    TAG_WIDTH, TAG_HEIGHT);
}

bool TrainsetGenerator::intersectsNone(const std::vector<cv::Rect> &tag_boxes,
                                       const cv::Rect & wrong_box) const {

    return std::all_of(
            tag_boxes.cbegin(), tag_boxes.cend(),
            [&wrong_box, this](const auto & box) {
                return (box & wrong_box).area() <= box.area()*max_intersection;
    });
}


int TrainsetGenerator::wrongAroundCoordinate() {
    int x = _around_wrong_dis(_random_gen);
    if(rand() % 2) {
        return x;
    } else {
        return -x;
    }
}
TrainsetGenerator TrainsetGenerator::operator=(TrainsetGenerator &&other) {
    return TrainsetGenerator(std::move(other));
}
void TrainsetGenerator::process(const std::vector<ImageDesc> &descs) {
    return process(descs.cbegin(), descs.cend());
}

void TrainsetGenerator::processParallel(const std::vector<ImageDesc> &img_descs) {
    using Iter = std::vector<ImageDesc>::const_iterator;
    std::vector<std::thread> threads;
    auto fn = std::mem_fn<void(Iter, Iter)>(&TrainsetGenerator::process<Iter>);

    _start_time = std::chrono::system_clock::now();
    _n_todo = img_descs.size();
    _n_done.store(0);
    printProgress(_start_time, 0);

    size_t n_threads = std::min(size_t(std::thread::hardware_concurrency()*2),
                             img_descs.size());
    size_t images_per_thread = img_descs.size() / n_threads;
    ASSERT(images_per_thread >= 1, "There are no lazy workers");
    for(size_t i = 0; i < n_threads && i < img_descs.size(); i++) {
        auto begin = img_descs.cbegin() + images_per_thread*i;
        auto end = img_descs.cbegin() + images_per_thread*(i+1);
        if (i + 1 == n_threads) {
            end = img_descs.cend();
        }
        threads.emplace_back(std::thread(fn, this, begin, end));
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
