
#include "TrainsetGenerator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>
#include <QTime>
#include <thread>

#include "utils.h"
#include "Dataset.h"
#include "DataWriter.h"


namespace deeplocalizer {

TrainsetGenerator::TrainsetGenerator() :
    TrainsetGenerator(std::make_unique<DevNullWriter>())
{}

TrainsetGenerator::TrainsetGenerator(
        std::unique_ptr<DataWriter> writer, double sample_rate,
        double scale, bool use_rotation, double acceptance_rate)

    :
    _random_gen(_rd()),
    _angle_dis(0, 360),
    _writer(std::move(writer)),
    _sample_rate(sample_rate),
    _scale(scale),
    _use_rotation(use_rotation),
    _acceptance_rate(acceptance_rate)
{}

TrainsetGenerator::TrainsetGenerator(TrainsetGenerator &&gen) :
    QObject(nullptr),
    _random_gen(_rd()),
    _angle_dis(std::move(gen._angle_dis)),
    _writer(std::move(gen._writer)) { }

cv::Mat TrainsetGenerator::randomAffineTransformation(const cv::Point2f & center) {
    static const double scale = 1;
    double angle = _angle_dis(_random_gen);
    qDebug() << angle;
    return cv::getRotationMatrix2D(center, angle, scale);
}

cv::Mat rotateSubimage(const cv::Mat & src,
                                           cv::Point2i center,
                                           double degrees) {
    cv::Mat subimage = getSubimage(src, tagBoxForCenter(center), TAG_WIDTH/2);
    cv::Mat rotated;
    int c = subimage.cols;
    int r = subimage.rows;
    int diagonal = int(sqrt(c*c + r*r));
    int offsetX = (diagonal - c) / 2;
    int offsetY = (diagonal - r) / 2;
    cv::Mat targetMat(diagonal, diagonal, src.type(), cv::Scalar(0));
    cv::Point2f target_center(targetMat.cols / 2.0f, targetMat.rows / 2.0f);
    subimage.copyTo(targetMat.rowRange(offsetY, offsetY + r)
                       .colRange(offsetX, offsetX + c));
    cv::Mat rot_mat = cv::getRotationMatrix2D(target_center, degrees, 1.0);
    cv::warpAffine(targetMat, rotated, rot_mat, targetMat.size());
    cv::Rect box(int(target_center.x - TAG_WIDTH / 2),
                 int(target_center.y - TAG_HEIGHT / 2),
                 TAG_WIDTH, TAG_HEIGHT);
    return rotated(box).clone();
}

std::vector<cv::Rect> getAllTagBoxes(const std::vector<Tag> & tags) {
    std::vector<cv::Rect>  boxes;
    for (auto & tag : tags) {
        if(tag.isTag()) {
            boxes.emplace_back(tag.getBoundingBox());
        }
    }
    return boxes;
}

void TrainsetGenerator::process(const ImageDesc &desc,
                                std::vector<TrainDatum> &train_data) {

    Image image(desc);
    const static int ndim = 2;
    cv::Mat image_mat = image.getCvMat();
    std::uniform_int_distribution<> x_dis(0, image_mat.cols);
    std::uniform_int_distribution<> y_dis(0, image_mat.rows);
    std::uniform_real_distribution<> acceptance_dis(0., 1.);

    auto generateRandomPoints = [&](const size_t n) {
        cv::Mat points(n, ndim, CV_32F);
        for(size_t i = 0; i < n; i++) {
            points.at<float>(i, 0) = static_cast<float>(x_dis(_random_gen));
            points.at<float>(i, 1) = static_cast<float>(y_dis(_random_gen));
        }
        return points;
    };

    auto boxes = getAllTagBoxes(desc.getTags());
    cv::Mat tag_centers(boxes.size(), ndim, CV_32F);
    for(size_t i = 0; i < boxes.size(); i++) {
        auto & box = boxes.at(i);
        tag_centers.at<float>(i, 0) = static_cast<float>(box.x + box.width/2);
        tag_centers.at<float>(i, 1) = static_cast<float>(box.y + box.height/2);
    }
    cv::flann::Index tags_index(tag_centers,
                                cv::flann::KDTreeIndexParams(1),
                                cvflann::FLANN_DIST_L2);

    auto neighborsDist = [&](const cv::Mat & points) {
        static const int k = 1;
        cv::Mat neighbors_idx; // (points.rows, k, CV_32S);
        cv::Mat dist; // (points.rows, k, CV_32F);
        tags_index.knnSearch(points, neighbors_idx, dist, k, cv::flann::SearchParams(64));
        return std::make_pair(neighbors_idx, dist);
    };

    const size_t start_size =  train_data.size();
    while(train_data.size() - start_size < _sample_rate*boxes.size()) {
        cv::Mat random_points =
                generateRandomPoints(desc.getTags().size());
        auto neighbors_dist = neighborsDist(random_points);
        auto dists = std::get<1>(neighbors_dist);
        for (int i = 0; i < random_points.rows; i++) {
            cv::Point2i center(static_cast<int>(random_points.at<float>(i, 0)),
                               static_cast<int>(random_points.at<float>(i, 1)));
            float d = dists.at<float>(i, 0);
            double tagness = exp(-0.5 * d / pow(28, 2));

            if (pow(tagness, 2) + acceptance_dis(_random_gen) >= 1 - _acceptance_rate) {
                cv::Mat subimage;
                cv::Rect box = tagBoxForCenter(center);
                double angle = 0.;
                if (_use_rotation && tagness >= 0.8) {
                    angle = _angle_dis(_random_gen);
                    subimage = rotateSubimage(image_mat, center, angle);

                } else {
                    subimage = getSubimage(image_mat, box);
                }

                std::stringstream description;
                description << image.filename() << "_" << center.x << "_" << center.y << "_" << tagness;
                train_data.emplace_back(TrainDatum(subimage, center, angle, tagness,
                                                   description.str()));
            }
        }
    }
}

TrainsetGenerator TrainsetGenerator::operator=(TrainsetGenerator &&other) {
    return TrainsetGenerator(std::move(other));
}

void TrainsetGenerator::postProcess(std::vector<TrainDatum> & data) const
{
    if (_scale == 1) {
        return;
    }
    for(size_t i = 0; i < data.size(); i++) {
        auto & datum = data.at(i);
        cv::Mat mat = datum.mat();
        cv::Mat scaledMat;
        cv::resize(mat, scaledMat, scaledMat.size(), _scale, _scale);
        TrainDatum scaled_datum(scaledMat,  datum.center(),
                                datum.rotation_angle(), datum.taginess());

        data.at(i) = scaled_datum;
    }
}

void TrainsetGenerator::processParallel(const std::vector<ImageDesc> &img_descs) {
    using Iter = std::vector<ImageDesc>::const_iterator;
    std::vector<std::thread> threads;
    auto fn = std::mem_fn<void(Iter, Iter)>(&TrainsetGenerator::generateAndWrite<Iter>);

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
