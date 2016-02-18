#ifndef DEEP_LOCALIZER_PROPOSALGENERATOR_H
#define DEEP_LOCALIZER_PROPOSALGENERATOR_H

#include <memory>
#include <QObject>
#include <QThread>

#include "Image.h"
#include "PipelineWorker.h"

namespace deeplocalizer {


class ProposalGenerator : public QObject {
    Q_OBJECT

public slots:
    void imageProcessed(ImageDesc image);
signals:
    void progress(double progress);
    void finished();

public:
    static const std::string IMAGE_DESC_EXT;

    explicit ProposalGenerator(const std::vector<std::string>& image_paths);
    explicit ProposalGenerator(const std::vector<ImageDesc>&  _image_desc);
    explicit ProposalGenerator(const std::vector<ImageDesc> & images_todo,
                               const std::vector<ImageDesc> & images_done);

    void processPipeline();

    inline const std::deque<ImageDesc> & getBeforePipelineImages() const {
        return _images_before_pipeline;
    }

    inline const std::deque<ImageDesc> & getProposalImages() const {
        return _images_with_proposals;
    }
    virtual ~ProposalGenerator();
private:
    explicit ProposalGenerator();
    void init();

    std::deque<ImageDesc> _images_before_pipeline;
    std::deque<ImageDesc> _images_with_proposals;
    unsigned long _n_images;

    std::vector<std::unique_ptr<PipelineWorker>> _workers;
    std::mutex _with_proposals_mutex;
};
}

#endif //DEEP_LOCALIZER_PROPOSALGENERATOR_H
