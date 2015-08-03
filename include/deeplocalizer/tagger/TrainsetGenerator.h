
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

using ImagePhasePair = std::pair<std::reference_wrapper<const ImageDesc>, Dataset::Phase>;
class TrainsetGenerator : public QObject  {
Q_OBJECT
public:
    TrainsetGenerator();
    TrainsetGenerator(TrainsetGenerator && gen);

    TrainsetGenerator(std::unique_ptr<DataWriter> writer);
    TrainsetGenerator operator=(TrainsetGenerator && other);
    ~TrainsetGenerator() = default;

    // number of rotated, translated samples per actual tag
    unsigned int samples_per_tag = 32;
    // number of wrong samples per actual tag
    unsigned int wrong_samples_per_tag = 32;
    // if positive allow wrong sample to reach into the bounding box of the tag
    const int shrinking = 30;

    unsigned long current_idx;
    boost::filesystem::path output_dir;

    cv::Mat randomAffineTransformation(const cv::Point2f & center);
    TrainDatum trainData(const ImageDesc & desc, const Tag & tag, const cv::Mat & subimage);
    void wrongSamplesAround(const Tag &tag,
                                               const ImageDesc &desc,
                                               const Image &img,
                                               std::vector<TrainDatum> &train_data);
    void wrongSamples(const ImageDesc & desc, std::vector<TrainDatum> & train_data);
    void trueSamples(const ImageDesc & desc, std::vector<TrainDatum> & train_data);
    void trueSamples(const ImageDesc & desc, const Tag &tag, const cv::Mat & subimage,
                     std::vector<TrainDatum> & train_data);
    cv::Mat rotate(const cv::Mat & src, double degrees);
    void process(const std::vector<ImageDesc> &descs);

    std::vector<ImagePhasePair>
    groupTestTrain(const std::vector<ImageDesc> & descs);

    template<typename InputIt>
    void process(InputIt begin, InputIt end) {
        for(InputIt iter = begin; iter != end; iter++) {
            std::vector<TrainDatum> data;
            ImagePhasePair elem = *iter;
            const ImageDesc & desc = elem.first;
            Dataset::Phase phase = elem.second;
            process(desc, data);
            _writer->write(data, phase);
            incrementDone();
        }
    }

    void processParallel(const std::vector<ImageDesc> &desc);
    void process(const ImageDesc & desc,
                 std::vector<TrainDatum> & train_data);
signals:
    void progress(double p);

private:
    static const int MIN_TRANSLATION;
    static const int MAX_TRANSLATION;
    static const int MIN_AROUND_WRONG;
    static const int MAX_AROUND_WRONG;
    std::random_device _rd;
    std::mt19937  _gen;
    std::uniform_real_distribution<double> _angle_dis;
    std::uniform_int_distribution<int> _translation_dis;
    std::uniform_int_distribution<int> _around_wrong_dis;
    std::unique_ptr<DataWriter> _writer;
    std::vector<cv::Rect> getNearbyTagBoxes(const Tag &tag,
                                            const ImageDesc &desc);
    cv::Rect proposeWrongBox(const Tag &tag);
    bool intersectsNone(std::vector<cv::Rect> &tag_boxes, cv::Rect wrong_box);
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
