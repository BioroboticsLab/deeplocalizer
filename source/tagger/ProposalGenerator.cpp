
#include "ProposalGenerator.h"

#include <thread>
#include "utils.h"

#include "serialization.h"

namespace deeplocalizer {


namespace io = boost::filesystem;

const std::string ProposalGenerator::IMAGE_DESC_EXT = "proposal.desc";

ProposalGenerator::ProposalGenerator() {
}

ProposalGenerator::ProposalGenerator(const std::vector<ImageDesc> & image_desc)
    : _images_before_pipeline(image_desc.cbegin(), image_desc.cend()),
      _n_images(image_desc.size())
{
    this->init();
}

ProposalGenerator::ProposalGenerator(
        const std::vector<ImageDesc> & images_todo,
        const std::vector<ImageDesc> & images_done
)
        : _images_before_pipeline(images_todo.cbegin(), images_todo.cend()),
          _images_with_proposals(images_done.cbegin(), images_done.cend()),
          _n_images(images_todo.size() + images_done.size())
{
    this->init();
}
ProposalGenerator::ProposalGenerator(const std::vector<std::string>& image_paths)
    : _n_images(image_paths.size())
{
    for (auto path : image_paths) {
        if (!io::exists(path)) {
            throw std::runtime_error("Could not open file");
        }
        _images_before_pipeline.push_back(ImageDesc(path));
    }
    this->init();
}

void ProposalGenerator::init() {
    // use only one worker. Because it seems that caffe can't handle multiple
    // threads, the relay on the pipeline for parallelism.
    int cpus = 1;
    for(int i = 0; i < (cpus != 0 ? cpus: 1); i++) {
        _workers.emplace_back(std::make_unique<PipelineWorker>());
    }
}


void ProposalGenerator::processPipeline() {
    size_t n = _workers.size();
    for(size_t i = 0; !_images_before_pipeline.empty(); i++) {
        size_t worker_idx = i % n;
        auto img = _images_before_pipeline.front();
        _images_before_pipeline.pop_front();
        auto & worker = _workers.at(worker_idx);
        PipelineWorker::tags_proposals_callback_t cb = std::bind(std::mem_fn(&ProposalGenerator::imageProcessed), this, std::placeholders::_1);
        worker->process(img, std::move(cb));
    }
    std::lock_guard<std::mutex> lk(_with_proposals_mutex);
    if (_images_with_proposals.size() == _n_images) {
        emit finished();
    }
}

void ProposalGenerator::imageProcessed(ImageDesc img) {
    std::lock_guard<std::mutex> lk(_with_proposals_mutex);
    _images_with_proposals.push_back(img);
    img.setSavePathExtension(IMAGE_DESC_EXT);
    img.save();
    emit progress(_images_with_proposals.size() / static_cast<double>(_n_images));
    if (_images_with_proposals.size() == _n_images) {
        emit finished();
    }
}

ProposalGenerator::~ProposalGenerator() {
    for(auto & worker: _workers) {
        worker->quit();
    }
}
}
