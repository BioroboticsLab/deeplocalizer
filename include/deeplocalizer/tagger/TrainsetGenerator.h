
#ifndef DEEP_LOCALIZER_TRAINSETGENERATOR_H
#define DEEP_LOCALIZER_TRAINSETGENERATOR_H

#include <boost/filesystem.hpp>
#include <QObject>

#include "Image.h"
#include "TrainDatum.h"
#include "Dataset.h"
#include "DataWriter.h"
#include "utils.h"

namespace deeplocalizer {

class TrainsetGenerator : public QObject  {
Q_OBJECT
public:
    static const int MIN_TRANSLATION;
    static const int MAX_TRANSLATION;
    static const int MIN_AROUND_WRONG;
    static const int MAX_AROUND_WRONG;
    static const double RATIO_AROUND_TO_UNIFORM_DEFAULT;
    static const double RATIO_TRUE_TO_FALSE_SAMPLES_DEFAULT;

    TrainsetGenerator();
    TrainsetGenerator(double ratio_around_uniform, double ratio_true_false);
    TrainsetGenerator(TrainsetGenerator && gen);

    TrainsetGenerator(double ratio_around_uniform, double ratio_true_false,
                      std::unique_ptr<DataWriter> writer);
    TrainsetGenerator operator=(TrainsetGenerator && other);
    ~TrainsetGenerator() = default;

    // number of rotated, translated samples per actual tag
    size_t samples_per_tag = 32;
    // number of wrong samples per actual tag
    size_t wrong_samples_per_tag = 32;
    // if positive allow wrong sample to reach into the bounding box of the tag
    const double max_intersection = 0.5;

    double scale = 1;

    unsigned long current_idx;
    boost::filesystem::path output_dir;

    cv::Mat randomAffineTransformation(const cv::Point2f & center);
    TrainDatum trainData(const ImageDesc & desc, const Tag & tag, const cv::Mat & subimage);
    void wrongSamplesAroundTag(const Tag &tag,
                            const ImageDesc &desc,
                            const Image &img,
                            std::vector<TrainDatum> &train_data);

    void wrongSamplesUniform(
                const ImageDesc &desc,  const Image &img,
                std::vector<TrainDatum> &train_data);
    void wrongSamples(const ImageDesc & desc, std::vector<TrainDatum> & train_data);
    void trueSamples(const ImageDesc & desc, std::vector<TrainDatum> & train_data);
    void trueSamples(const ImageDesc & desc, const Tag &tag, const cv::Mat & subimage,
                     std::vector<TrainDatum> & train_data);
    cv::Mat rotate(const cv::Mat & src, double degrees);
    void process(const std::vector<ImageDesc> &descs);

    void postProcess(std::vector<TrainDatum> & data) const;

    template<typename InputIt>
    void process(InputIt begin, InputIt end) {
        std::vector<TrainDatum> data;
        for(InputIt iter = begin; iter != end; iter++) {
            const ImageDesc & desc = *iter;
            process(desc, data);
            postProcess(data);
            _writer->write(data);
            incrementDone();
            data.clear();
        }
    }

    void processParallel(const std::vector<ImageDesc> &desc);
    void process(const ImageDesc & desc,
                 std::vector<TrainDatum> & train_data);
    void wrongSamplesAroundTag(const ImageDesc &desc, const Image &img, std::vector<TrainDatum> &train_data);
    void wrongSamplesAround(const ImageDesc &desc, const Image &img, std::vector<TrainDatum> &train_data);
signals:
    void progress(double p);

private:
    std::random_device _rd;
    std::mt19937  _random_gen;
    std::uniform_real_distribution<double> _angle_dis;
    std::uniform_int_distribution<int> _translation_dis;
    std::uniform_int_distribution<int> _around_wrong_dis;
    double _avg_samples_per_tag;
    double _samples_around_err;
    // ratio of around-tag samples to uniform samples
    double _ratio_around_uniform;
    // ratio of true samples to false samples
    double _ratio_true_false;
    std::unique_ptr<DataWriter> _writer;
    std::vector<cv::Rect> getNearbyTagBoxes(const Tag &tag,
                                            const ImageDesc &desc);
    cv::Rect proposeWrongBoxAround(const Tag &tag);
    cv::Rect proposeWrongBoxUniform(const cv::Mat & mat);
    bool intersectsNone(const std::vector<cv::Rect> &tag_boxes, const cv::Rect & wrong_box) const ;
    void wrongSampleRot90(const Image &img,
                          const cv::Rect &wrong_box,
                          std::vector<TrainDatum> &train_data);
    int wrongAroundCoordinate();
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    std::atomic_uint _n_done;
    unsigned long _n_todo;
    void incrementDone();
};
}

#endif //DEEP_LOCALIZER_TRAINSETGENERATOR_H
