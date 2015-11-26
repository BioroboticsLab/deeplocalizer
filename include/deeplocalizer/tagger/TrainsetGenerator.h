
#ifndef DEEP_LOCALIZER_TRAINSETGENERATOR_H
#define DEEP_LOCALIZER_TRAINSETGENERATOR_H

#include <boost/filesystem.hpp>
#include <QObject>

#include "Image.h"
#include "TrainDatum.h"
#include "Dataset.h"
#include "DataWriter.h"
#include "utils.h"
#include "deeplocalizer_tagger.h"

namespace deeplocalizer {

class TrainsetGenerator : public QObject  {
Q_OBJECT
public:
    TrainsetGenerator();
    TrainsetGenerator(TrainsetGenerator && gen);

    TrainsetGenerator(std::unique_ptr<DataWriter> writer, double sample_rate=32,
                      double scale=1., bool use_rotation=true);
    TrainsetGenerator operator=(TrainsetGenerator && other);
    ~TrainsetGenerator() = default;

    unsigned long current_idx;
    boost::filesystem::path output_dir;

    cv::Mat randomAffineTransformation(const cv::Point2f & center);
    TrainDatum trainData(const ImageDesc & desc, const Tag & tag, const cv::Mat & subimage);

    cv::Mat rotate(const cv::Mat & src, double degrees);

    void inline generateAndWrite(const std::vector<ImageDesc> &descs) {
        return generateAndWrite(descs.cbegin(), descs.cend());
    }
    void postProcess(std::vector<TrainDatum> & data) const;

    template<typename InputIt>
    void generateAndWrite(InputIt begin, InputIt end) {
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
signals:
    void progress(double p);

private:
    std::random_device _rd;
    std::mt19937  _random_gen;
    std::uniform_real_distribution<double> _angle_dis;
    std::unique_ptr<DataWriter> _writer;
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    // number of samples per actual tag
    double _sample_rate = 32;
    double _scale = 1;
    bool _use_rotation = true;
    std::atomic_uint _n_done;
    unsigned long _n_todo;
    void incrementDone();
};
}

#endif //DEEP_LOCALIZER_TRAINSETGENERATOR_H
